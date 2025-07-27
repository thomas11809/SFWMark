import os
import itertools
from tqdm import tqdm
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from utils import *

# main 함수
def main(args):
    set_random_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / f"{args.output_dir}/{args.dataset_id}/{args.wm_type}"
    os.makedirs(os.path.join(save_dir, "img_pil"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img_pil_wm"), exist_ok=True)

    # [Datasets]
    meta_annot, prompt_key, gt_folder = get_text_dataset(args.dataset_id)

    # [Evaluation Settings]
    num_dataset = len(meta_annot)
    RANGE_EVAL = range(0,num_dataset)
    w_seed_list = [*range(w_seed, w_seed + wm_capacity)] # 2048 seed numbers
    identify_gt_indices = np.random.choice(wm_capacity, size=num_dataset).tolist()
    np.save(os.path.join(save_dir, f"identify_gt_indices_{num_dataset}.npy"), identify_gt_indices)
    
    # [Stable-Diffusion-v2-1-base Settings]
    model_id = "stabilityai/stable-diffusion-2-1-base"
    resolution = 512
    torch_dtype = torch.float32

    # [Load Stable-Diffusion pipeline]
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # [Make GT patterns] wm_capacity=2048
    if args.wm_type == "Tree-Ring":
        masks = tree_masks
        Fourier_watermark_pattern_list = [make_Fourier_treering_pattern(pipe, shape, this_w_seed) for this_w_seed in w_seed_list]
    elif args.wm_type == "RingID":
        # Following the official RingID implementation
        masks = ringid_masks
        single_channel_num_slots = RADIUS - RADIUS_CUTOFF # int(math.log2(wm_capacity))
        key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-64, 64, 2).tolist(), repeat=len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
        key_value_combinations = list(itertools.product(*key_value_list))
        Fourier_watermark_pattern_list = [make_Fourier_ringid_pattern(pipe, shape, list(combo), w_seed=w_seed_list[i],
            radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
            ring_watermark_channel=RING_WATERMARK_CHANNEL, heter_watermark_channel=HETER_WATERMARK_CHANNEL,
            heter_watermark_region_mask=heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL)>0 else None)
            for i, combo in enumerate(key_value_combinations)]
        # A. fix_gt (from official implementation)
        Fourier_watermark_pattern_list = [fft(ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in Fourier_watermark_pattern_list]
        # B. time_shift (from official implementation)
        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim=(-1, -2)))
    elif args.wm_type == "HSTR":
        masks = tree_masks
        masks[:, HETER_WATERMARK_CHANNEL] = single_channel_heter_watermark_mask # (64,64) RounderRingMask for Hetero Watermark (noise)
        Fourier_watermark_pattern_list = [make_Fourier_treering_pattern(pipe, shape, this_w_seed, 
            hs=True, center=True, heter=True) for this_w_seed in w_seed_list]
    elif args.wm_type == "HSQR":
        assert box_size == 2
        Fourier_watermark_pattern_list = [make_hsqr_pattern(idx=this_w_seed) for this_w_seed in w_seed_list]
    assert len(Fourier_watermark_pattern_list) == wm_capacity
    
    # [Save Fourier_watermark_pattern_list]
    torch.save(torch.stack(Fourier_watermark_pattern_list, 0).detach(), os.path.join(save_dir, f"pattern_list-{wm_capacity}.pt"))

    print("Generation Starts")
    batch_size = 8
    for batch_start in tqdm(range(0, len(RANGE_EVAL), batch_size)):
        batch_indices = RANGE_EVAL[batch_start:batch_start+batch_size]
        batch_size_actual = len(batch_indices) # N
        # File inputs
        gen_prompts = [meta_annot[idx][prompt_key] for idx in batch_indices]
        file_names = [f"{idx}.png" for idx in batch_indices]
        # Set random seeds
        set_random_seed(42 + batch_start)

        with torch.no_grad():
            key_indices = [identify_gt_indices[key] for key in batch_indices]
            pattern_gt_batch = [Fourier_watermark_pattern_list[key_index] for key_index in key_indices]
            # adjust dims of pattern_gt_batch
            if len(pattern_gt_batch[0].shape) == 4:
                pattern_gt_batch = torch.cat(pattern_gt_batch, dim=0) # (N,4,64,64) for Tree-Ring, RingID, HSTR
            elif len(pattern_gt_batch[0].shape) == 3:
                pattern_gt_batch = torch.stack(pattern_gt_batch, dim=0) # (N,c_wm,42,42) for HSQR
            else:
                raise ValueError(f"Unexpected pattern_gt_batch shape: {pattern_gt_batch[0].shape}")
            assert len(pattern_gt_batch.shape) == 4

            # get random latents ~ N(0,I)
            no_watermark_latents = get_random_latents(pipe, batch_size=batch_size_actual) # (N,4,64,64)
            # watermark injection
            if args.wm_type in ["Tree-Ring", "RingID"]:
                Fourier_watermark_latents, _ = inject_wm(no_watermark_latents, pattern_gt_batch, masks, cut_real=True, device=device)
            elif args.wm_type == "HSTR":
                Fourier_watermark_latents, _ = inject_wm(no_watermark_latents, pattern_gt_batch, masks, center=True, cut_real=False, device=device)
            elif args.wm_type == "HSQR":
                Fourier_watermark_latents = inject_hsqr(no_watermark_latents, pattern_gt_batch, center=True, device=device)
            
            # generate images
            batched_latents = torch.cat([no_watermark_latents, Fourier_watermark_latents], dim=0) # (2N,4,64,64)
            generated_images = pipe(gen_prompts*2, latents=batched_latents, guidance_scale=7.5,
                num_inference_steps=50, num_images_per_prompt=1).images
            
            # [Free GPU Memory]
            torch.cuda.empty_cache()
        
        # Save images
        img_pils, img_pil_wms = generated_images[:batch_size_actual], generated_images[batch_size_actual:]
        for i, idx in enumerate(batch_indices):
            img_pils[i].save(os.path.join(save_dir, f"img_pil/{file_names[i]}"))
            img_pil_wms[i].save(os.path.join(save_dir, f"img_pil_wm/{file_names[i]}"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--wm_type", choices=["Tree-Ring", "RingID", "HSTR", "HSQR"], required=True, help="Choose semantic watermarking methods following merged-in-generation scheme")
    parser.add_argument("--dataset_id", choices=["coco", "Gustavo", "DB1k"], required=True, help="Choose dataset_id")
    parser.add_argument("--output_dir", default="outputs", help="output directory: ./[output_dir]/")
    args = parser.parse_args()
    main(args)
    