import numpy as np
import os
from sklearn import metrics
from prettytable import PrettyTable

case_names = ["Clean", "Brightness", "Contrast", "JPEG", "Blur", "Noise", "BM3D",
              "VAE-B", "VAE-C", "Diff", "CC", "RC", "Avg"]

datasets = ["coco", "Gustavo", "DB1k"]
wm_types_semantic = ["Tree-Ring", "zodiac", "HSTR", "RingID", "HSQR"]
wm_types_bit = ["dwtDct", "dwtDctSvd", "rivaGan", "ssig"]

results_all = {}

for dataset_id in datasets:
    results = {metric: {} for metric in ["AUC", "MaxAcc", "TPR@1%FPR", "Id-Acc", # Semantic WM
                                        "BitAcc", "Perfect-Match-Rate"]} # Bitstream-based WM
    
    for wm_type in wm_types_semantic:
        dst_dir = f"results/{dataset_id}/{wm_type}"
        verify_l1_npz = np.load(os.path.join(dst_dir, "verify-l1.npz"))
        identify_acc_npz = np.load(os.path.join(dst_dir, "identify-acc.npz"))
        no_wms = verify_l1_npz["no_wm"]
        wms = verify_l1_npz["wm"]
        i_wms = identify_acc_npz["wm"]
        
        auc_list, acc_list, low_list = [], [], []
        for idx in range(no_wms.shape[1]): # 12 Cases
            no_wm = no_wms[:, idx].tolist()
            wm = wms[:, idx].tolist()
            distances = no_wm + wm
            labels = [0] * len(no_wm) + [1] * len(wm)
            fpr, tpr, _ = metrics.roc_curve(labels, distances, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            acc = 1 - ((fpr + (1 - tpr)) / 2).min()
            low = tpr[np.where(fpr < 0.01)[0][-1]] if np.any(fpr < 0.01) else 0.0
            
            auc_list.append(auc)
            acc_list.append(acc)
            low_list.append(low)
        
        id_accs = np.mean(i_wms.squeeze(), axis=0).tolist()
        
        for metric_name, values in zip(
            ["AUC", "MaxAcc", "TPR@1%FPR", "Id-Acc"],
            [auc_list, acc_list, low_list, id_accs]
        ):
            results[metric_name][wm_type] = values + [np.mean(values)]
    
    for wm_type in wm_types_bit:
        dst_dir = f"results/{dataset_id}/{wm_type}"
        if wm_type == "ssig":
            wm_diff_total = np.load(os.path.join(dst_dir, "wm_diff_total.npy"))  # (1000,12,48)
            wm_diff_total = np.mean(wm_diff_total, axis=-1)  # (1000,12)
            ba_total = np.mean(wm_diff_total, axis=0)
            ia_total = np.mean(wm_diff_total == 1, axis=0)
        else:
            ba_total = np.load(os.path.join(dst_dir, "bit-acc-total.npy"))  # (1000,12)
            ia_total = np.mean(ba_total == 1, axis=0)
            ba_total = np.mean(ba_total, axis=0)
        
        for metric_name, values in zip(
            ["BitAcc", "Perfect-Match-Rate"],
            [ba_total, ia_total]
        ):
            results[metric_name][wm_type] = values.tolist() + [np.mean(values)]
    
    results_all[dataset_id] = results


print("#" * 60)
print("Table 1: Verification Performance")
table = PrettyTable()
table.field_names = ["WM Type"] + case_names
for dataset_id in datasets:
    results = results_all[dataset_id]
    for wm_type in wm_types_bit:
        table.add_row([wm_type] + [f"{v:.3f}" for v in results["BitAcc"][wm_type]])
    for wm_type in wm_types_semantic:
        table.add_row([wm_type] + [f"{v:.3f}" for v in results["TPR@1%FPR"][wm_type]])
    if dataset_id != "DB1k":
        table.add_row(["-" * 5 for col in table.field_names])
print(table)
print()

print("#" * 60)
print("Table 2: Identification Accuracy")
table = PrettyTable()
table.field_names = ["WM Type"] + case_names
for dataset_id in datasets:
    results = results_all[dataset_id]
    for wm_type in wm_types_bit:
        table.add_row([wm_type] + [f"{v:.3f}" for v in results["Perfect-Match-Rate"][wm_type]])
    for wm_type in wm_types_semantic:
        table.add_row([wm_type] + [f"{v:.3f}" for v in results["Id-Acc"][wm_type]])
    if dataset_id != "DB1k":
        table.add_row(["-" * 5 for col in table.field_names])
print(table)
print()

print("#" * 60)
print("Table 11: [Bit Accuracy] Unified Detection Performance")
table = PrettyTable()
table.field_names = ["WM Type"] + case_names
for dataset_id in datasets:
    results = results_all[dataset_id]
    for wm_type in wm_types_bit:
        table.add_row([wm_type] + [f"{v:.3f}" for v in results["BitAcc"][wm_type]])
    for wm_type in wm_types_semantic:
        table.add_row([wm_type] + [f"{v:.3f}" for v in results["Id-Acc"][wm_type]])
    if dataset_id != "DB1k":
        table.add_row(["-" * 5 for col in table.field_names])
print(table)
print()

print("#" * 60)
print("Table 12: [Verification] AUC values for Semantic Methods")
table = PrettyTable()
table.field_names = ["WM Type"] + case_names
for dataset_id in datasets:
    results = results_all[dataset_id]
    for wm_type in wm_types_semantic:
        table.add_row([wm_type] + [f"{v:.3f}" for v in results["AUC"][wm_type]])
    if dataset_id != "DB1k":
        table.add_row(["-" * 5 for col in table.field_names])
print(table)
print()

print("#" * 60)
print("Table 13: [Verification] Maximum Accuracy for Semantic Methods")
table = PrettyTable()
table.field_names = ["WM Type"] + case_names
for dataset_id in datasets:
    results = results_all[dataset_id]
    for wm_type in wm_types_semantic:
        table.add_row([wm_type] + [f"{v:.3f}" for v in results["MaxAcc"][wm_type]])
    if dataset_id != "DB1k":
        table.add_row(["-" * 5 for col in table.field_names])
print(table)
print()

