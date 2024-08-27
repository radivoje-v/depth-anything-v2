import argparse
import os
import random
import cv2

import torch
import torchvision.transforms as T
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

import numpy as np
import logging
from afe.apis.defines import default_quantization
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.defines import InputName
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source
from sima_utils.data.data_generator import DataGenerator
from afe.apis.error_handling_variables import enable_verbose_error_messages


parser = argparse.ArgumentParser(description='Depth Anything V2 Model Evaluation')


parser.add_argument('--model_name', type=str, default='inference_depth_anything_v2_vits_opt_modified.onnx')
parser.add_argument('--output_dir', type=str, default='./models/inference_depth_anything_v2_vits_kitti_real')
parser.add_argument('--calib_dir', type=str, default=None)

args = parser.parse_args()

MODEL_DIR = './models'
OUTPUT_DIR = args.output_dir

models = [args.model_name,
          ]


cal_dir = args.calib_dir


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



# Function to load a random subset of images from the val directory
def load_calibration_data(root_dir):
    image_files = os.listdir(root_dir)
    # random.shuffle(image_files)  # Shuffle the list of image files
    calibration_data = []
    for img_file in image_files:
        img_path = os.path.join(root_dir, img_file)
        image = cv2.imread(img_path)


        input_tensor, (h, w) = image2tensor(image, input_size=518)
        image_np = np.array(input_tensor.cpu())
        transposed_image = np.transpose(image_np, (0, 2, 3, 1))

        calibration_data.append(transposed_image)

    cal_images = np.concatenate(calibration_data, axis=0)

    inputs = {'input': cal_images}

    calibration_data = convert_data_generator_to_iterable(DataGenerator(inputs))

    return calibration_data



def compile_model(model_name: str, arm_only: bool):
    print(f"Compiling model {model_name} with arm_only={arm_only}")
    enable_verbose_error_messages()

    # Models importer parameters
    input_name, input_shape, input_type = ("input", (1, 3, 518, 518), ScalarType.float32)
    input_shapes_dict = {input_name: input_shape}
    input_types_dict = {input_name: input_type}

    model_path = os.path.join(MODEL_DIR, f"{model_name}")
    importer_params = onnx_source(model_path, input_shapes_dict, input_types_dict)

    model_prefix = os.path.splitext(model_path)[0]
    output_dir = os.path.join(OUTPUT_DIR, f"{model_prefix}")
    os.makedirs(output_dir, exist_ok=True)
    loaded_net = load_model(importer_params)

    if not cal_dir:
        inputs = {InputName(input_name): np.random.rand(1, 518, 518, 3)}
        dg = DataGenerator(inputs)
        calibration_data = convert_data_generator_to_iterable(dg)
    else:
        calibration_data = load_calibration_data(cal_dir)

    model_sdk_net = loaded_net.quantize(calibration_data,
                                        default_quantization,
                                        model_name=model_name,
                                        arm_only=arm_only,
                                        log_level=logging.INFO)

    saved_model_directory = OUTPUT_DIR
    model_sdk_net.save(model_name=model_name, output_directory=saved_model_directory)

    model_sdk_net.compile(output_path=output_dir,
                          retained_temporary_directory_name=output_dir,
                          batch_size=1,
                          log_level=logging.INFO)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model in models:
        compile_model(model, False)


if __name__ == "__main__":
    main()
