import os
from tqdm import tqdm
from pathlib import Path
import torch
from attdiffusion import DiffWMAttacker, ReSDPipeline
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("--wm_type", required=True, help="Choose watermarking type")
parser.add_argument("--dataset_id", choices=["coco", "Gustavo", "DB1k"], required=True, help="Choose dataset_id")
parser.add_argument("--output_dir", default="outputs", help="output directory: ./[output_dir]/")
args = parser.parse_args()

project_root = Path(__file__).resolve().parent.parent.parent
save_dir = project_root / f"{args.output_dir}/{args.dataset_id}/{args.wm_type}"
os.makedirs(os.path.join(save_dir, "img_pil-diffatt_fp16"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "img_pil_wm-diffatt_fp16"), exist_ok=True)

# [Attack Settings]
batch_size = 8
detect_trials = 1000
RANGE_EVAL = range(0,detect_trials)

# [Load Diffusion-Attacker pipeline]
att_pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
att_pipe.set_progress_bar_config(disable=True)
att_pipe.to("cuda")
attacker = DiffWMAttacker(att_pipe, batch_size=batch_size, noise_step=60, captions={})

# [Attack Loop]
for batch_start in tqdm(range(0, len(RANGE_EVAL), batch_size)):
    batch_indices = RANGE_EVAL[batch_start:batch_start+batch_size]
    file_names = [f"{idx}.png" for idx in batch_indices]

    # 1. Attack - no_wm_img
    input_paths = [os.path.join(save_dir, "img_pil", file_name) for file_name in file_names]
    output_paths = [os.path.join(save_dir, "img_pil-diffatt_fp16", file_name) for file_name in file_names]
    attacker.attack(input_paths, output_paths)
    
    # 2. Attack - wm_img
    input_paths = [os.path.join(save_dir, "img_pil_wm", file_name) for file_name in file_names]
    output_paths = [os.path.join(save_dir, "img_pil_wm-diffatt_fp16", file_name) for file_name in file_names]
    attacker.attack(input_paths, output_paths)
