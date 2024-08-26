#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
gpus=1
encoder=vits
dataset=vkitti # hypersim
img_size=518
min_depth=0.001
max_depth=20
#model_path=../metric_depth_anything_v2_vits_opt_kitti
#model_path=../models/simplified_metric_depth_anything_v2_vits_opt_modified.onnx
model_path=../checkpoints/depth_anything_v2_metric_vkitti_vits.pth
save_path=exp/vkitti/val # exp/hypersim
mkdir -p $save_path
python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    val.py --encoder $encoder --save-path $save_path --dataset $dataset \
    --img_size $img_size --min-depth $min_depth --max-depth $max_depth --model-path $model_path \
    --port 20596 2>&1 | tee -a $save_path/$now.log
