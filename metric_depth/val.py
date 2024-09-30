import argparse
import logging
import os
import pprint
import sys
import warnings

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from util.dist_helper import setup_distributed
from util.metric import eval_depth
from util.utils import init_log
import onnxruntime as ort
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet



os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

parser = argparse.ArgumentParser(description='Depth Anything V2 Model Evaluation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti'])
parser.add_argument('--img_size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--quantized_model_name', type=str, default='inference_depth_anything_v2_vits_opt_modified.onnx')
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--sample_size', default=-1, type=int)
parser.add_argument('--metric', default=0, type=int)


def create_onnx_session(model_path):
    # Create an ONNX Runtime session with GPU support
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

    # Log available providers and ensure CUDAExecutionProvider is available
    available_providers = session.get_providers()
    print(f"Available providers: {available_providers}")
    if 'CUDAExecutionProvider' not in available_providers:
        raise RuntimeError("CUDAExecutionProvider not available. Ensure that ONNXRuntime-GPU is installed and CUDA is set up correctly.")

    return session


def postprocess_onnx(pred, max_depth):
    # DepthAnythingV2 from metric_depth/depth_anything_v2/dpt.py (train + val model arch) last layer is Sigmoid
    # (compared to Relu + Identity in DepthAnythingV2 from depth_anything_v2/dpt.py (inference arch))
    # and model output is multiplied with max_depth argument
    pred = pred.squeeze(0)
    pred = torch.from_numpy(pred)
    pred = F.sigmoid(pred)
    return pred * max_depth



def main():
    args = parser.parse_args()

    warnings.simplefilter('ignore', np.RankWarning)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model_p = args.model_path
    if args.model_path.endswith(".pth"):
        rank, world_size = setup_distributed(port=args.port)

        if rank == 0:
            all_args = {**vars(args), 'ngpus': world_size}
            logger.info('{}\n'.format(pprint.pformat(all_args)))

        local_rank = int(os.environ["LOCAL_RANK"])

    else:
        all_args = {**vars(args)}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size, sample_size=args.sample_size)
    else:
        raise NotImplementedError

    if args.model_path.endswith(".pth"):
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)

    else:
        valloader =  DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


    if args.model_path.endswith(".pth"):
        extension = "pth"

        if args.metric:
            from depth_anything_v2.dpt import DepthAnythingV2
            model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

        else:
            parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            sys.path.insert(0, parent_dir)

            from depth_anything_v2.dpt import DepthAnythingV2
            model = DepthAnythingV2(**{**model_configs[args.encoder]})

        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        model.to(DEVICE).eval()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                          output_device=local_rank, find_unused_parameters=True)
    elif args.model_path.endswith(".onnx"):
        extension = "onnx"

        session = create_onnx_session(args.model_path)
    elif os.path.isdir(args.model_path):

        from afe.apis.model import Model
        from afe.ir.defines import InputName

        extension = "quant"

        quant_model_path = args.model_path
        quantized_model = Model.load(args.quantized_model_name, quant_model_path)
    else:
        raise ValueError(f"Wrong file extension for model path: {args.model_path}. Please provide model in .pth, .onnx or SiMa-quantized format")

    results = {'d1': torch.tensor([0.0]).to(DEVICE), 'd2': torch.tensor([0.0]).to(DEVICE), 'd3': torch.tensor([0.0]).to(DEVICE),
               'abs_rel': torch.tensor([0.0]).to(DEVICE), 'sq_rel': torch.tensor([0.0]).to(DEVICE), 'rmse': torch.tensor([0.0]).to(DEVICE),
               'rmse_log': torch.tensor([0.0]).to(DEVICE), 'log10': torch.tensor([0.0]).to(DEVICE), 'silog': torch.tensor([0.0]).to(DEVICE)}
    nsamples = torch.tensor([0.0]).to(DEVICE)

    for i, sample in enumerate(valloader):
        print(f"Sample: {i}/{len(valloader)}")

        img, depth, valid_mask = sample['image'].to(DEVICE).float(), sample['depth'].to(DEVICE)[0], sample['valid_mask'].to(DEVICE)[0]

        if extension == "pth":
            with torch.no_grad():
                pred = model(img)
        elif extension == "onnx":
            image = np.array(img.cpu())

            ort_inputs = {session.get_inputs()[0].name: image}
            pred = session.run(None, ort_inputs)[0]
            if len(pred.shape) == 4:
                pred = pred.squeeze(0)

            pred = np.transpose(pred, (2,0,1)) # bcs we modified output for MLA
            pred = torch.from_numpy(pred)
        elif extension == "quant":
            input_image = np.array(img.cpu())
            transposed_image = np.transpose(input_image, (0, 2, 3, 1))  # NCHW -> NHWC
            pred = quantized_model.execute({InputName('input'): transposed_image})[0]
            pred = np.transpose(pred, (0,3,1,2)) # NHWC -> NCHW

            if len(pred.shape) == 4:
                pred = pred.squeeze(0)

            pred = np.transpose(pred, (2, 0, 1)) # bcs we modified output for MLA
            pred = torch.from_numpy(pred)

        pred = pred.to(DEVICE)

        pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

        if valid_mask.sum() < 10:
            continue

        cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

    if args.model_path.endswith(".pth"):

        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

    if not args.model_path.endswith(".pth") or rank == 0:
        # logger.info('==========================================================================================')
        # logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        # logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
        # logger.info('==========================================================================================')
        print('==========================================================================================')
        print('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        print('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
        print('==========================================================================================')

if __name__ == '__main__':
    main()