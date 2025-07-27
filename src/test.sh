#!/bin/bash
# [coco]
python generate.py --wm_type Tree-Ring --dataset_id coco
python generate.py --wm_type RingID --dataset_id coco
python generate.py --wm_type HSTR --dataset_id coco
python generate.py --wm_type HSQR --dataset_id coco

python metric.py --wm_type Tree-Ring --dataset_id coco
python metric.py --wm_type RingID --dataset_id coco
python metric.py --wm_type HSTR --dataset_id coco
python metric.py --wm_type HSQR --dataset_id coco

python diff_attack/diff_wm_attack.py --wm_type Tree-Ring --dataset_id coco
python diff_attack/diff_wm_attack.py --wm_type RingID --dataset_id coco
python diff_attack/diff_wm_attack.py --wm_type HSTR --dataset_id coco
python diff_attack/diff_wm_attack.py --wm_type HSQR --dataset_id coco

python detect.py --wm_type Tree-Ring --dataset_id coco
python detect.py --wm_type RingID --dataset_id coco
python detect.py --wm_type HSTR --dataset_id coco
python detect.py --wm_type HSQR --dataset_id coco

# [Gustavo]
python generate.py --wm_type Tree-Ring --dataset_id Gustavo
python generate.py --wm_type RingID --dataset_id Gustavo
python generate.py --wm_type HSTR --dataset_id Gustavo
python generate.py --wm_type HSQR --dataset_id Gustavo

python metric.py --wm_type Tree-Ring --dataset_id Gustavo
python metric.py --wm_type RingID --dataset_id Gustavo
python metric.py --wm_type HSTR --dataset_id Gustavo
python metric.py --wm_type HSQR --dataset_id Gustavo

python diff_attack/diff_wm_attack.py --wm_type Tree-Ring --dataset_id Gustavo
python diff_attack/diff_wm_attack.py --wm_type RingID --dataset_id Gustavo
python diff_attack/diff_wm_attack.py --wm_type HSTR --dataset_id Gustavo
python diff_attack/diff_wm_attack.py --wm_type HSQR --dataset_id Gustavo

python detect.py --wm_type Tree-Ring --dataset_id Gustavo
python detect.py --wm_type RingID --dataset_id Gustavo
python detect.py --wm_type HSTR --dataset_id Gustavo
python detect.py --wm_type HSQR --dataset_id Gustavo

# [DB1k]
python generate.py --wm_type Tree-Ring --dataset_id DB1k
python generate.py --wm_type RingID --dataset_id DB1k
python generate.py --wm_type HSTR --dataset_id DB1k
python generate.py --wm_type HSQR --dataset_id DB1k

python metric.py --wm_type Tree-Ring --dataset_id DB1k
python metric.py --wm_type RingID --dataset_id DB1k
python metric.py --wm_type HSTR --dataset_id DB1k
python metric.py --wm_type HSQR --dataset_id DB1k

python diff_attack/diff_wm_attack.py --wm_type Tree-Ring --dataset_id DB1k
python diff_attack/diff_wm_attack.py --wm_type RingID --dataset_id DB1k
python diff_attack/diff_wm_attack.py --wm_type HSTR --dataset_id DB1k
python diff_attack/diff_wm_attack.py --wm_type HSQR --dataset_id DB1k

python detect.py --wm_type Tree-Ring --dataset_id DB1k
python detect.py --wm_type RingID --dataset_id DB1k
python detect.py --wm_type HSTR --dataset_id DB1k
python detect.py --wm_type HSQR --dataset_id DB1k