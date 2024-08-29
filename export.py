import argparse
import logging
import os
import pprint
import warnings

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

# from dataset.hypersim import Hypersim
# from dataset.kitti import KITTI
from depth_anything_v2.dpt import DepthAnythingV2 as InferenceModel
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as MetricModel
from metric_depth.util.dist_helper import setup_distributed
# from metric_depth.util.metric import eval_depth
from metric_depth.util.utils import init_log
import onnxruntime as ort
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

parser = argparse.ArgumentParser(description='Depth Anything V2 Model exporting')

parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--img_size', default=518, type=int)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    pth = args.checkpoint

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = MetricModel(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    model.load_state_dict(torch.load(pth, map_location='cpu'))
    model.to(DEVICE).eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(DEVICE)

    output_path = f"{pth[:-4]}_metric_exported.onnx"
    torch.onnx.export(model, dummy_input, output_path,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None,
                      opset_version=17)

    print(f"Metric model exported to {output_path} successfully.")

    depth_anything = InferenceModel(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(pth, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    output_path = f"{pth[:-4]}_inference_exported.onnx"
    torch.onnx.export(depth_anything, dummy_input, output_path,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None,
                      opset_version=17)

    print(f"Inference model exported to {output_path} successfully.")


if __name__ == '__main__':
    main()