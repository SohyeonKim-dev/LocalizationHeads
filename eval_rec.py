#!/usr/bin/env python3
"""REC (Acc@0.5) evaluation with fixed localization heads on lmms-lab/RefCOCO.

Loads the model once, runs one forward per referring expression, takes the
attention of the last input token to the 576 visual tokens, combines the fixed
heads and turns the result into a bounding box.

Box variants computed from the same attention:
  paper          : Gaussian(sigma=1, k=7) -> sum -> > mean (whole padded 24x24 grid)
                   -> largest 8-connected component -> grid box mapped back through
                   LLaVA-1.5 square padding
  paper_cropmean : same as paper, but the mean is taken over cells inside the original
                   image only (the paper does not say whether padding is included)
  paper_imgres   : combined map bilinearly upsampled to the padded square (S x S), cropped
                   to the original W x H, > mean of that crop, largest 8-connected
                   component -> tight pixel box (paper does not state the resolution)
  repo           : public code as-is (bbox.py: truncate=4 smoothing, all pixels, no pad fix)

Diagnostic only (reported separately, not a method):
  oracle_thr     : paper pipeline, but per sample the threshold is the one (out of
                   ORACLE_QUANTILES of the combined map) with the best GT IoU.
                   Upper bound of what fixing the threshold alone could reach.
  pointing       : the peak cell of the combined map, mapped to the image, lies inside
                   the GT box (does the attention point at the target at all?).

Example:
  python eval_rec.py --split val --limit 200 --viz 8
  python eval_rec.py --split testA
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from lab.stations import MetadataStation
from bbox import combine_heads, binarize_mean_relu, bbox_from_mask, scale_bbox_to_image

# LLaVA-1.5-7B localization heads (paper Fig. 6a / Fig. 7), (layer, head)
DEFAULT_HEADS = "14-24,14-13,14-26"
VARIANTS = ["paper", "paper_cropmean", "paper_imgres", "repo"]
ORACLE_QUANTILES = np.linspace(0.02, 0.98, 25)


def parse_heads(s):
    return [tuple(int(v) for v in item.split("-")) for item in s.split(",")]


def build_input_ids(query, tokenizer, model, conv_mode):
    # Same prompt construction as collector.collect_attention
    assert not model.config.mm_use_im_start_end
    qs = DEFAULT_IMAGE_TOKEN + "\n" + query
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", conv=conv)
    return input_ids.unsqueeze(0).to(model.device)


@torch.inference_mode()
def last_token_attn_to_vis(model, input_ids, image_tensor, image_size):
    """Returns [L, H, V] float32 cpu attention from the last token to the visual tokens."""
    out = model(
        input_ids=input_ids,
        images=image_tensor.unsqueeze(0),  # 5D like collector.py; the 4D path in llava_arch is broken
        image_sizes=[image_size],
        output_attentions=True,
        return_dict=True,
    )
    b = MetadataStation.get_begin_pos("vis")
    v = MetadataStation.get_vis_len()
    attn = torch.stack([a[0, :, -1, b:b + v] for a in out.attentions], dim=0)  # [L, H, V]
    return attn.float().cpu()


def largest_component(mask):
    lab, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    return (lab == (1 + int(np.argmax(sizes)))).astype(np.uint8)


def grid_box_to_image(box, P, W, H, pad_fix):
    """Inclusive grid box (x0, y0, x1, y1) -> continuous image xyxy."""
    x0, y0, x1, y1 = box
    if pad_fix:
        # expand2square: original pasted at offset ((S-W)//2, (S-H)//2) inside an S x S square
        S = max(W, H)
        ox, oy = (S - W) // 2, (S - H) // 2
        c = S / P
        bx = [x0 * c - ox, y0 * c - oy, (x1 + 1) * c - ox, (y1 + 1) * c - oy]
    else:
        bx = [x0 * W / P, y0 * H / P, (x1 + 1) * W / P, (y1 + 1) * H / P]
    return [float(np.clip(bx[0], 0, W)), float(np.clip(bx[1], 0, H)),
            float(np.clip(bx[2], 0, W)), float(np.clip(bx[3], 0, H))]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def valid_cells(P, W, H):
    """Boolean [P, P]: grid cells that overlap the original image inside the padded square."""
    S = max(W, H)
    ox, oy = (S - W) // 2, (S - H) // 2
    edges = np.arange(P + 1) * S / P
    cols = (edges[1:] > ox) & (edges[:-1] < ox + W)
    rows = (edges[1:] > oy) & (edges[:-1] < oy + H)
    return rows[:, None] & cols[None, :]


def box_at_threshold(M, thr, P, W, H):
    cc = largest_component((M > thr).astype(np.uint8))
    return grid_box_to_image(bbox_from_mask(cc), P, W, H, pad_fix=True), cc


def peak_in_gt(M, P, W, H, gt):
    """Peak cell center of the combined map (through square padding) inside the GT box."""
    r, c = np.unravel_index(int(np.argmax(M)), M.shape)
    S = max(W, H)
    x = (c + 0.5) * S / P - (S - W) // 2
    y = (r + 0.5) * S / P - (S - H) // 2
    return gt[0] <= x <= gt[2] and gt[1] <= y <= gt[3]


def oracle_iou(M, valid, P, W, H, gt):
    """Best GT IoU over thresholds at quantiles of the combined map (inside the image)."""
    best = 0.0
    for thr in np.quantile(M[valid], ORACLE_QUANTILES):
        best = max(best, iou(box_at_threshold(M, thr, P, W, H)[0], gt))
    return best


def imgres_box(M, W, H):
    """Upsample the grid map to the padded square, crop to the image, binarize at its mean."""
    S = max(W, H)
    ox, oy = (S - W) // 2, (S - H) // 2
    up = F.interpolate(torch.from_numpy(M)[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0].numpy()
    crop = up[oy:oy + H, ox:ox + W]
    cc = largest_component((crop > crop.mean()).astype(np.uint8))
    x0, y0, x1, y1 = bbox_from_mask(cc)
    return [float(x0), float(y0), float(x1 + 1), float(y1 + 1)]


def predict_boxes(attn, heads, W, H):
    """attn: [L, H, V]. Returns (dict variant -> xyxy, combined map, valid-cell mask)."""
    V = attn.shape[-1]
    P = int(round(V ** 0.5))
    maps = [attn[l, h].reshape(P, P).numpy() for l, h in heads]

    # Paper: Gaussian k=7, sigma=1 (truncate=3 -> radius 3), element-wise sum, > mean
    M = sum(gaussian_filter(m, sigma=1.0, truncate=3.0) for m in maps)
    valid = valid_cells(P, W, H)

    boxes = {
        "paper": box_at_threshold(M, M.mean(), P, W, H)[0],
        "paper_cropmean": box_at_threshold(M, M[valid].mean(), P, W, H)[0],
        "paper_imgres": imgres_box(M, W, H),
    }

    # Public code path (bbox.py), unchanged; its pixel box is inclusive -> +1 for xyxy
    attn4 = attn[:, :, None, :]
    sel = [{"layer": l, "head": h} for l, h in heads]
    M_repo = combine_heads(attn4, sel, P=P, sigma=1.0)
    rx0, ry0, rx1, ry1 = scale_bbox_to_image(bbox_from_mask(binarize_mean_relu(M_repo)), (W, H), P)
    boxes["repo"] = [float(rx0), float(ry0), float(rx1 + 1), float(ry1 + 1)]
    return boxes, M, valid


def save_viz(path, image, M, gt, boxes, expr, accs):
    W, H = image.size
    S = max(W, H)
    ox, oy = (S - W) // 2, (S - H) // 2
    # Heatmap rendered in the padded square, then cropped back to the original frame
    m = (M - M.min()) / (M.max() - M.min() + 1e-8)
    heat = Image.fromarray((m * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR).crop((ox, oy, ox + W, oy + H))
    heat = np.asarray(heat, dtype=np.float32)[..., None] / 255.0
    base = np.asarray(image, dtype=np.float32)
    red = np.zeros_like(base); red[..., 0] = 255
    over = Image.fromarray((base * (1 - 0.5 * heat) + red * 0.5 * heat).astype(np.uint8))

    canvas = Image.new("RGB", (2 * W, H + 40), "white")
    canvas.paste(image, (0, 0)); canvas.paste(over, (W, 0))
    d = ImageDraw.Draw(canvas)
    colors = {"gt": "lime", "paper": "red", "repo": "cyan"}
    for x in (0, W):
        d.rectangle([gt[0] + x, gt[1], gt[2] + x, gt[3]], outline=colors["gt"], width=3)
        d.rectangle([boxes["paper"][0] + x, boxes["paper"][1], boxes["paper"][2] + x, boxes["paper"][3]], outline=colors["paper"], width=3)
        d.rectangle([boxes["repo"][0] + x, boxes["repo"][1], boxes["repo"][2] + x, boxes["repo"][3]], outline=colors["repo"], width=1)
    d.text((5, H + 4), f"{expr!r}  {W}x{H}", fill="black")
    d.text((5, H + 20), f"GT=green paper=red(IoU {accs['paper']:.2f}) repo=cyan(IoU {accs['repo']:.2f})", fill="black")
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test", "testA", "testB"])
    ap.add_argument("--limit", type=int, default=-1, help="number of objects (rows); -1 = all")
    ap.add_argument("--heads", default=DEFAULT_HEADS, help="comma-separated layer-head, e.g. 14-24,14-13,14-26")
    ap.add_argument("--no-period", action="store_true", help="do not append '.' to expressions")
    ap.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--conv-mode", default="referseg")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/rec_eval")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--viz", type=int, default=0, help="save first N visualizations (+ first portrait/landscape)")
    ap.add_argument("--save-attn", action="store_true", help="save [L,H,V] fp16 attention per expression (~1.2MB each)")
    args = ap.parse_args()

    heads = parse_heads(args.heads)
    run_name = args.run_name or f"{args.split}_n{args.limit}" + ("_noperiod" if args.no_period else "")
    out_dir = os.path.join(args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)
    if args.viz:
        os.makedirs(os.path.join(out_dir, "viz"), exist_ok=True)
    if args.save_attn:
        os.makedirs(os.path.join(out_dir, "attn"), exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("lmms-lab/RefCOCO", split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model, cache_dir=None, model_base=None,
        model_name=get_model_name_from_path(args.model), device=args.device,
    )
    model.eval()

    correct = {k: 0 for k in VARIANTS}
    area_ratios = {k: [] for k in VARIANTS}
    oracle_correct = 0
    pointing_correct = 0
    n_total, n_viz, seen_shape = 0, 0, set()
    t0 = time.time()
    with open(os.path.join(out_dir, "preds.jsonl"), "w") as fout:
        for ri, row in enumerate(ds):
            image = row["image"].convert("RGB")
            W, H = image.size
            x, y, w, h = row["bbox"]
            gt = [x, y, x + w, y + h]
            image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

            for ei, expr in enumerate(row["answer"]):
                query = expr.strip()
                if not args.no_period and not query.endswith("."):
                    query = query + "."
                input_ids = build_input_ids(query, tokenizer, model, args.conv_mode)
                attn = last_token_attn_to_vis(model, input_ids, image_tensor, (W, H))
                boxes, M, valid = predict_boxes(attn, heads, W, H)
                ious = {k: iou(boxes[k], gt) for k in VARIANTS}
                for k in VARIANTS:
                    correct[k] += int(ious[k] >= 0.5)
                    area_ratios[k].append(box_area(boxes[k]) / box_area(gt))
                o_iou = oracle_iou(M, valid, int(round(attn.shape[-1] ** 0.5)), W, H, gt)
                oracle_correct += int(o_iou >= 0.5)
                hit = peak_in_gt(M, int(round(attn.shape[-1] ** 0.5)), W, H, gt)
                pointing_correct += int(hit)
                n_total += 1

                rec = {
                    "question_id": row["question_id"], "expr_idx": ei, "query": query,
                    "last_token": tokenizer.convert_ids_to_tokens(int(input_ids[0, -1])),
                    "image_size": [W, H], "gt_xyxy": gt, "boxes": boxes, "iou": ious, "oracle_iou": o_iou, "pointing_hit": bool(hit),
                }
                fout.write(json.dumps(rec) + "\n")

                if args.save_attn:
                    torch.save(attn.half(), os.path.join(out_dir, "attn", f"{row['question_id']}_{ei}.pt"))

                shape = "portrait" if H > W else "landscape" if W > H else "square"
                if args.viz and ei == 0 and (n_viz < args.viz or shape not in seen_shape):
                    seen_shape.add(shape)
                    n_viz += 1
                    save_viz(os.path.join(out_dir, "viz", f"{ri:05d}_{row['question_id']}_{shape}.jpg"),
                             image, M, gt, boxes, query, ious)

            if (ri + 1) % 50 == 0:
                accs = " ".join(f"{k}={100 * correct[k] / n_total:.1f}" for k in VARIANTS)
                accs += f" | oracle_thr={100 * oracle_correct / n_total:.1f} pointing={100 * pointing_correct / n_total:.1f}"
                print(f"[{ri + 1}/{len(ds)}] exprs={n_total} {accs} ({time.time() - t0:.0f}s)", flush=True)

    summary = {
        "split": args.split, "num_objects": len(ds), "num_expressions": n_total,
        "heads": heads, "append_period": not args.no_period,
        "acc@0.5": {k: 100 * correct[k] / max(n_total, 1) for k in VARIANTS},
        "median_area_pred_over_gt": {k: float(np.median(area_ratios[k])) for k in VARIANTS},
        "diagnostic_upper_bound": {"oracle_thr_acc@0.5": 100 * oracle_correct / max(n_total, 1),
                                   "num_quantiles": len(ORACLE_QUANTILES),
                                   "pointing_acc": 100 * pointing_correct / max(n_total, 1)},
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
