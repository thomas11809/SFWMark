import os
import numpy as np
from prettytable import PrettyTable

results = {metric: {} for metric in ["FID", "CLIP"]}

dataset_id = "coco"
wm_semantic_merged_in_gen = ["Tree-Ring", "RingID", "HSTR", "HSQR"]
for wm_type in wm_semantic_merged_in_gen:
    dst_dir = f"results/{dataset_id}/{wm_type}"
    fid_npz = np.load(os.path.join(dst_dir, "fid.npz"))
    clip_npz = np.load(os.path.join(dst_dir, "clip.npz"))

    results["FID"][wm_type] = fid_npz["img_pil_wm"]
    results["CLIP"][wm_type] = np.mean(clip_npz["img_pil_wm"])

print("#" * 60)
print("Table 4: [Generative Quality] for Semantic Methods following the merged-in-generation scheme")
table = PrettyTable()
table.field_names = ["WM Type", "FID", "CLIP"]
for wm_type in wm_semantic_merged_in_gen:
    table.add_row([wm_type, f"{results['FID'][wm_type]:.3f}", f"{results['CLIP'][wm_type]:.3f}"])
print(table)
print()