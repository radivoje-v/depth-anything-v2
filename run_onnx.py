import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


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

def image2tensor(raw_image, input_size=518):
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    h, w = raw_image.shape[:2]

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    image = transform({'image': image})['image']

    image = torch.from_numpy(image).unsqueeze(0)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    image = image.to(DEVICE)

    return image, (h, w)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--img_path', type=str)
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--model_path', type=str, default='./models/simplified_depth_anything_v2_small_518_518_opt_modified.onnx')
    parser.add_argument('--quantized_model_name', type=str, default='inference_depth_anything_v2_vits_opt_modified.onnx')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    model_path = args.model_path
    # session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    if args.model_path.endswith(".onnx"):
        extension = "onnx"

        model_name = os.path.basename(args.model_path[:-5])
        outdir = os.path.join(args.outdir, model_name)
        session = create_onnx_session(args.model_path)
    elif os.path.isdir(args.model_path):

        from afe.apis.model import Model
        from afe.ir.defines import InputName

        model_name = os.path.basename(args.model_path)
        outdir = os.path.join(args.outdir, model_name)

        extension = "quant"
        quant_model_path = args.model_path
        quantized_model = Model.load(args.quantized_model_name, quant_model_path)
    else:
        raise ValueError(f"Wrong file extension for model path: {args.model_path}. Please provide model in .onnx or SiMa-quantized format")

    os.makedirs(outdir, exist_ok=True)

    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)

        input_tensor, (h, w) = image2tensor(raw_image, input_size=518)
        image = np.array(input_tensor.cpu())

        if extension == "onnx":

            image = cv2.resize(np.transpose(np.squeeze(image, axis=0), (1,2,0)), (args.img_size, args.img_size))
            image = np.expand_dims(np.transpose(image, (2,0,1)), axis=0)

            ort_inputs = {session.get_inputs()[0].name: image}
            pred = session.run(None, ort_inputs)[0]

            if len(pred.shape) == 4:
                pred = pred.squeeze(0)

            pred = np.transpose(pred, (2,0,1)) # bcs we modified output for MLA
            pred = torch.from_numpy(pred)
        elif extension == "quant":

            image = cv2.resize(np.transpose(np.squeeze(image, axis=0), (1,2,0)), (args.img_size, args.img_size))
            image = np.expand_dims(np.transpose(image, (2,0,1)), axis=0)
            input_image = image

            transposed_image = np.transpose(input_image, (0, 2, 3, 1))
            pred = quantized_model.execute({InputName('input'): transposed_image})
            pred = pred[0].transpose(0, 3, 1, 2)

            if len(pred.shape) == 4:
                pred = pred.squeeze(0)

            pred = np.transpose(pred, (2,0,1)) # bcs we modified output for MLA
            pred = torch.from_numpy(pred)

        depth = pred
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        depth = depth.cpu().numpy()

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        resulting_file_name = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + f'_{model_name}.png')

        if args.pred_only:
            cv2.imwrite(resulting_file_name, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(resulting_file_name, combined_result)

        print(f"Result saved to {resulting_file_name}")