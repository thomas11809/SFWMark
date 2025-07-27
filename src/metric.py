import os
import itertools
from tqdm import tqdm
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import open_clip

from utils import *

# main 함수
def main(args):
    set_random_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / f"{args.output_dir}/{args.dataset_id}/{args.wm_type}"

    # [Datasets]
    meta_annot, prompt_key, gt_folder = get_text_dataset(args.dataset_id)

    # [Evaluation Settings]
    detect_trials = 1000
    RANGE_EVAL = range(0,detect_trials)
    
    # 1. CLIP
    # [Load CLIP model]
    reference_model = "ViT-g-14"
    reference_model_pretrain = "laion2b_s12b_b42k"
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(reference_model, pretrained=reference_model_pretrain, device=device)
    ref_tokenizer = open_clip.get_tokenizer(reference_model)

    # [Get CLIP scores] text-image alignment
    clip_values = {"img_pil": [], "img_pil_wm": []}
    batch_size = 64
    for batch_start in tqdm(range(0, len(RANGE_EVAL), batch_size)):
        batch_indices = RANGE_EVAL[batch_start:batch_start+batch_size]
        # File inputs
        text_prompts = [meta_annot[idx][prompt_key] for idx in batch_indices]
        # get batched CLIP scores
        img_pils_batch = [Image.open(os.path.join(save_dir, f"img_pil/{idx}.png")) for idx in batch_indices]
        img_pils_wm_batch = [Image.open(os.path.join(save_dir, f"img_pil_wm/{idx}.png")) for idx in batch_indices]
        clip_values["img_pil"].append(get_clip_score(img_pils_batch, text_prompts, ref_model, ref_clip_preprocess, ref_tokenizer, device=device).cpu().numpy()) # (N,)
        clip_values["img_pil_wm"].append(get_clip_score(img_pils_wm_batch, text_prompts, ref_model, ref_clip_preprocess, ref_tokenizer, device=device).cpu().numpy()) # (N,)
    
    # Save CLIP results
    clip_values["img_pil"] = np.concatenate(clip_values["img_pil"])
    clip_values["img_pil_wm"] = np.concatenate(clip_values["img_pil_wm"])
    np.savez(os.path.join(save_dir, "clip.npz"), **clip_values)
    # Print CLIP results
    img_pil_clip_mean = np.mean(clip_values["img_pil"])
    img_pil_wm_clip_mean = np.mean(clip_values["img_pil_wm"])
    print("=" * 50)
    print(f"CLIP img_pil : {img_pil_clip_mean:.3f}")
    print(f"CLIP img_pil_wm : {img_pil_wm_clip_mean:.3f}")
    
    # 2. FID
    # [FID Settings] only coco has reference images (gt_folder)
    if args.dataset_id == "coco":
        fid_values = {}
        fid_values["img_pil"] = get_FID(gt_folder, os.path.join(save_dir, "img_pil"), device=device)
        fid_values["img_pil_wm"] = get_FID(gt_folder, os.path.join(save_dir, "img_pil_wm"), device=device)

        # Save FID results
        np.savez(os.path.join(save_dir, "fid.npz"), **fid_values)
        # Print FID results
        print("=" * 50)
        print(f'FID img_pil: {fid_values["img_pil"]:.3f}')
        print(f'FID img_pil_wm: {fid_values["img_pil_wm"]:.3f}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--wm_type", required=True, help="Choose watermarking methods")
    parser.add_argument("--dataset_id", choices=["coco", "Gustavo", "DB1k"], required=True, help="Choose dataset_id")
    parser.add_argument("--output_dir", default="outputs", help="output directory: ./[output_dir]/")
    args = parser.parse_args()
    main(args)
    