import argparse
from pathlib import Path
from typing import List, Dict, Optional

import os
import numpy as np
import onnx
import onnx.numpy_helper
from numba.core.cgutils import printf
from onnxsim import simplify

from onnx import helper, numpy_helper

import cv2
import urllib

import sys

from sima_utils.onnx import onnx_helpers as oh

_MIDLE_BLOCKS = 3


parser = argparse.ArgumentParser(description="Graph surgery for DepthAnythingV2")

parser.add_argument('--val', dest='val', action='store_true',
                    help='if provided will do surgery for val model; otherwise for inference model')
parser.add_argument('--model', type=str, help='relative path to the onnx model',
                    default="./models/depth_anything_v2_vits.onnx")
parser.add_argument('--k', type=int, default=2)

args = parser.parse_args()


_source_onnx_fname = args.model

_onnx_path = Path(_source_onnx_fname)

simplified_fname = f"{args.model[:-5]}_simplified.onnx"

model_to_be_simplified = oh.load_model(_source_onnx_fname)
model_simp, check = simplify(model_to_be_simplified)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, simplified_fname)

_source_onnx_fname = Path(args.model)
_simplified_onnx_fname = Path(simplified_fname)

# _input_shape = (1, 3, 518, 518)
_input_shape = (1, 3, 518, 518)
_model_input_name = "input"
_model_output_name = "output"

# Input and output of transformer blocks
_block_input_name = "/Add_output_0"
_block_output_name = f"/blocks.2/Add_1_output_0"

# Top-level splits
_preproc_split = oh.ModelSplit("preproc", [_model_input_name], [_block_input_name],
                               parent_path=_simplified_onnx_fname)
_transformer_split = oh.ModelSplit("vit", [_block_input_name], [_block_output_name],
                                   parent_path=_simplified_onnx_fname)
_postproc_split = oh.ModelSplit("postproc", [_block_output_name], [_model_output_name],
                                parent_path=_simplified_onnx_fname)

_top_level_splits: List[oh.ModelSplit] = [_preproc_split, _transformer_split, _postproc_split]

val = args.val

def _run_model(model_name, input_names, input_data):
    import onnxruntime as ort

    assert len(input_names) == len(input_data)

    sess = ort.InferenceSession(str(model_name))
    outputs = sess.run([], {name: np_data for name, np_data in zip(input_names, input_data)})
    return outputs

def _generate_test_vector_no_shape():
    import torchvision.transforms as T

    def _imread_url(u, readFlag=cv2.IMREAD_COLOR):

        resp = urllib.request.urlopen(u)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, readFlag)
        return image

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    print(f"Downloading {url}...")
    im = _imread_url(url)

    print(f"Done.\n Image shape = {im.shape[:2]}")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize([_input_shape[2], _input_shape[3]]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(im).unsqueeze(0)
    return img.numpy()


def _run_model(model_name, input_names, input_data):
    import onnxruntime as ort

    assert len(input_names) == len(input_data)

    sess = ort.InferenceSession(str(model_name))
    outputs = sess.run([], {name: np_data for name, np_data in zip(input_names, input_data)})
    return outputs


def _replace_node(model, node, new_nodes):
    for i, x in enumerate(list(model.graph.node)):
        if x.name != node.name:
            continue
        model.graph.node.remove(node)

        for j, new_node in enumerate(new_nodes):
            model.graph.node.insert(i + j, new_node)
        return


def make_einsums(output, node, equation, num):
    einsums = []
    for i in range(num):
        einsum_i = onnx.helper.make_node(
            name=node.name + "__" +str(i), op_type="Einsum",
            inputs=[output[i], node.input[1]],
            outputs=[node.name + "_" +str(i) +"_out"],
            equation=equation
        )
        einsums.append(einsum_i)
    return einsums


def make_softmaxs(node, einsum_output, num):
    softmaxs = []
    for i in range(num):
        softmax_i = onnx.helper.make_node(
            name=node.name + "_" +str(i), op_type=node.op_type,
            inputs=[einsum_output[i]],
            outputs=[node.name + "_" +str(i) +"_out"],
            axis=1
        )
        softmaxs.append(softmax_i)
    return softmaxs

# Transformer splits
def _split_transformer(blocks):
    """
    Block 0, 1, 2, ..., 8
    0: inputs = /Add_output_0, 		      part-B inputs = /blocks.0/Add_output_0, outputs = /blocks.0/Add_1_output_0
    1: inputs = /blocks.0/Add_1_output_0, part-B inputs = /blocks.1/Add_output_0, outputs = /blocks.1/Add_1_output_0
    2: inputs = /blocks.1/Add_1_output_0, part-B inputs = /blocks.2/Add_output_0, outputs = /blocks.2/Add_1_output_0
    3: inputs = /blocks.2/Add_1_output_0, part-B inputs = /blocks.3/Add_output_0, outputs = /blocks.3/Add_1_output_0
    ...
    8: inputs = /blocks.8/Add_1_output_0,	part-B inputs = /blocks.8/Add_output_0,	outputs = /blocks.8/Add_1_output_0
    """
    blocks_attn = []
    blocks_mlp = []
    for n in range(blocks):
        a_input = f"/blocks.{n - 1}/Add_1_output_0" if n >= 1 else "/Add_output_0"
        b_input = f"/blocks.{n}/Add_output_0"
        output = f"/blocks.{n}/Add_1_output_0"

        attn = oh.ModelSplit(f"block.{n}_attn", [a_input], [b_input], parent_split=_transformer_split)
        mlp = oh.ModelSplit(f"block.{n}_mlp", [b_input], [output], parent_split=_transformer_split)
        blocks_attn.append(attn)
        blocks_mlp.append(mlp)

    return blocks_attn, blocks_mlp


def _split_transformer_post(start, stop):
    """
    Block 9, 10, 11
    9: inputs = /blocks.9/Add_1_output_0, part-B inputs = /blocks.9/Add_output_0, outputs = /blocks.9/Add_1_output_0
    10: inputs = /blocks.10/Add_1_output_0, part-B inputs = /blocks.10/Add_output_0, outputs = /blocks.10/Add_1_output_0
    11: inputs = /blocks.11/Add_1_output_0, part-B inputs = /blocks.11/Add_output_0, outputs = /blocks.11/Add_1_output_0
    """
    blocks_attn = []
    blocks_mlp = []
    for n in range(start, stop):
        a_input = f"/blocks.{n - 1}/Add_1_output_0" if n >= 1 else "/Add_output_0"
        b_input = f"/blocks.{n}/Add_output_0"
        output = f"/blocks.{n}/Add_1_output_0"

        attn = oh.ModelSplit(f"block.{n}_attn", [a_input], [b_input], parent_split=_postproc_split)
        mlp = oh.ModelSplit(f"block.{n}_mlp", [b_input], [output], parent_split=_postproc_split)
        blocks_attn.append(attn)
        blocks_mlp.append(mlp)

    return blocks_attn, blocks_mlp


transformer_blocks_attn: List[oh.ModelSplit] = []
transformer_blocks_mlp: List[oh.ModelSplit] = []


def _generate_test_vector(shape):
    np.random.seed(123)
    data = np.random.uniform(-1, 1, shape).astype(np.float32)
    return data


def _modify_preproc():
    preproc_fname = _preproc_split.filename
    model = onnx.load(preproc_fname)
    graph_def = model.graph
    initializers = graph_def.initializer

    # Get weight shape in (M, C/group, kH, kW)
    [w_tensor] = [t for t in initializers if t.name == "pretrained.patch_embed.proj.weight"]
    w = onnx.numpy_helper.to_array(w_tensor)
    M, C_G, kH, kW = w.shape

    # Compute output height of convolution
    oH = (_input_shape[2] - kH) // kH + 1

    # Rewrite nodes
    for node_idx, node in enumerate(graph_def.node):
        if "Conv" in node.name:
            dilations = (node.attribute[0].ints[0], node.attribute[0].ints[1])
            group = node.attribute[1].i
            pads = (node.attribute[3].ints[0], node.attribute[3].ints[1],
                    node.attribute[3].ints[2], node.attribute[3].ints[3])
            strides = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            # Check this convolution is linear projection
            assert group == 1 and pads == (0, 0, 0, 0) and dilations == (1, 1)
            assert strides == (kH, kW)

        elif "Reshape" in node.name:
            model.graph.node.remove(node)

            # Insert a slice node per row
            conv_output = "/patch_embed/proj/Conv_output_0"
            slice_axes = "slice_axes"
            initializers.append(
                onnx.helper.make_tensor(slice_axes, onnx.TensorProto.INT64, [1], [2])
            )
            split_outputs = []
            for h in range(oH):
                slice_starts = f"slice_{h}_starts"
                slice_ends = f"slice_{h}_ends"
                initializers.append(
                    onnx.helper.make_tensor(slice_starts, onnx.TensorProto.INT64, [1], [h])
                )
                initializers.append(
                    onnx.helper.make_tensor(slice_ends, onnx.TensorProto.INT64, [1], [h + 1])
                )
                output = f"/patch_embed/Slice_{h}"
                slice_node = onnx.helper.make_node(
                    op_type="Slice",
                    name=f"/patch_embed/Slice_{h}",
                    inputs=[conv_output, slice_starts, slice_ends, slice_axes],
                    outputs=[output])
                model.graph.node.insert(node_idx + h, slice_node)
                split_outputs.append(output)

        elif "Transpose" in node.name:
            model.graph.node.remove(node)

            # Insert a concat node
            model.graph.node.insert(
                node_idx,
                onnx.helper.make_node(
                    op_type="Concat",
                    name="/patch_embed/rows",
                    inputs=split_outputs,
                    outputs=["/pre_reshaped_rows"],
                    axis=3
                )
            )

            # Hack to keep last Concat + Add as NHWC layout
            # So, insert a layout transform here
            model.graph.node.insert(
                node_idx + 1,
                onnx.helper.make_node(
                    op_type="Transpose",
                    name="/pre/to_nhwc",
                    inputs=["/pre_reshaped_rows"],
                    outputs=node.output,  # ["/patch_embed/Transpose_output_0"],
                    perm=(0, 2, 3, 1)
                )
            )

        elif "Concat" in node.name:
            model.graph.node.remove(node)
            model.graph.node.insert(
                node_idx,
                onnx.helper.make_node(
                    op_type="Concat",
                    name=node.name,
                    inputs=[node.input[0], node.input[1]],  # ["/Expand_output_0", "/patch_embed/Transpose_output_0"],
                    outputs=node.output,  # ["/Concat_output_0"],
                    axis=2  # NHWC layout, W axis
                )
            )

        elif "Add" in node.name:
            # Append a transpose to get back NCHW
            model.graph.node.remove(node)
            model.graph.node.insert(
                node_idx,
                onnx.helper.make_node(
                    op_type="Add",
                    name=node.name,
                    inputs=node.input,  # Keep input of Add
                    outputs=["/Add_output_nhwc"]
                )
            )
            model.graph.node.insert(
                node_idx + 1,
                onnx.helper.make_node(
                    op_type="Transpose",
                    name="/pre/to_nchw",
                    inputs=["/Add_output_nhwc"],
                    outputs=node.output,  # Keep output name of Add
                    perm=(0, 3, 1, 2)
                )
            )

    # Update constants
    # Class embedding: "/Expand_output_0", (1, 1, 384) -> (1, 1, 1, 384)
    # Position embedding: "/Cast_output_0", (1, 257, 384) -> (1, 1, 257, 384)
    for initializer in list(initializers):
        if initializer.name in ["581"]:
            graph_def.initializer.remove(initializer)
        elif initializer.name in ["/Expand_output_0", "/Cast_output_0"]:
            data = onnx.numpy_helper.to_array(initializer)
            # data = data.transpose([0, 2, 1])
            data = np.expand_dims(data, axis=1)
            initializer.CopyFrom(onnx.numpy_helper.from_array(data, initializer.name))

    # Update model output
    oh.update_io_shape(model, "/Add_output_0", (1, 384, 1, 1370))

    # Remove existing shape inference results.
    for value_info in list(model.graph.value_info):
        model.graph.value_info.remove(value_info)

    model, check = simplify(model)
    assert check, "Opt for preproc: Simplified ONNX model can not be validated"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_onnx_fname = Path(preproc_fname.stem + "_opt.onnx")
    onnx.save(model, output_onnx_fname)

    print(f'Opt for preproc ONNX file saved to: {output_onnx_fname}')
    return output_onnx_fname


def _verify_opt_preproc(src_model, opt_model, input_names, input_data):
    ref_outputs = oh.run_model(src_model, input_names, input_data)
    new_outputs = oh.run_model(opt_model, input_names, input_data)

    opt_outputs = []
    for no in new_outputs:
        t = np.squeeze(no, axis=2).transpose((0, 2, 1))
        opt_outputs.append(t)

    diff = ref_outputs[0] - opt_outputs[0]
    print(f"Difference: min = {np.min(diff)}, max = {np.max(diff)}")

    assert np.array_equal(ref_outputs, opt_outputs)
    print(f"Verification OK - opt model {opt_model} is equal to {src_model}")


def _modify_attn_post(idx, model, transformer_blocks_attn, start):
    idx_pos = idx - start
    # Remove nodes
    part = "attn"
    remove_list = _make_remove_list(idx_pos, part, _ops_to_be_removed[part])
    oh.remove_nodes_by_name_list(model, remove_list)

    # Insert transpose pair
    at_node = f"/blocks.{idx}/norm1/LayerNormalization"
    perm_before = (0, 2, 3, 1)
    perm_after = (0, 3, 1, 2)
    new_transpose_nodes = oh.insert_transpose_pair(model, at_node, perm_before, perm_after)

    # 2x Replace MatMul with Conv
    at_nodes = [
        f"/blocks.{idx}/attn/qkv/MatMul",
        f"/blocks.{idx}/attn/proj/MatMul"
    ]
    new_conv_nodes = oh.rewrite_matmul_as_conv(model, at_nodes)

    # 2x Replace MatMul with Einsum
    equations = {
        f"/blocks.{idx}/attn/MatMul": "nchw,nchq->nqhw",
        f"/blocks.{idx}/attn/MatMul_1": "nchw,nqhc->nqhw"
    }
    new_einsum_nodes = oh.rewrite_matmul_as_einsum(model, equations)

    # Insert slices
    after_node = f"/blocks.{idx}/attn/qkv/Add"
    slices_after_add = oh.insert_slices_after_node(model, after_node, axis=1, nslices=3, slice_size=384)

    # 3x Insert slices-concat
    nodes_after_concat = [
        f"/blocks.{idx}/attn/Mul",
        new_einsum_nodes[0],
        new_einsum_nodes[1]
    ]
    new_concat_nodes = []
    for i in range(3):
        new_node = oh.insert_slices_concat_between_nodes(model, slices_after_add[i], nodes_after_concat[i],
                                                         slice_axis=1, nslices=6, slice_size=64,
                                                         concat_axis=2)
        new_concat_nodes.append(new_node)

    # 1x Insert slices-concat
    after_node = new_einsum_nodes[1]
    before_node = new_conv_nodes[1]
    new_node = oh.insert_slices_concat_between_nodes(model, after_node, before_node,
                                                     slice_axis=2, nslices=6, slice_size=1,
                                                     concat_axis=1)
    new_concat_nodes.append(new_node)

    # Update I/O shape: (1, 257, 384) -> (1, 384, 1, 257)
    if idx_pos == 0:
        oh.update_io_shape(model, transformer_blocks_attn[idx_pos].input_names[0], (1, 384, 1, 1370))

    # Update constants
    node = oh.find_node(model, f"/blocks.{idx}/attn/qkv/Add")
    param_name = node.input[0]  # "blocks.0.attn.qkv.bias"
    perm = None
    new_shape = (1, 1152, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)
    node = oh.find_node(model, f"/blocks.{idx}/attn/proj/Add")
    param_name = node.input[0]  # "blocks.0.attn.proj.bias"
    perm = None
    new_shape = (1, 384, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)
    node = oh.find_node(model, f"/blocks.{idx}/ls1/Mul")
    param_name = node.input[1]  # "blocks.0.ls1.gamma"
    perm = None
    new_shape = (1, 384, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)

    # Update attributes
    oh.set_attribute_to_node(model, f"/blocks.{idx}/attn/Softmax", "axis", 1)

    # Connect newly created nodes
    oh.connect_nodes(model, [new_transpose_nodes[1], new_conv_nodes[0]], 0, 0)
    oh.connect_nodes(model, [new_conv_nodes[0], f"/blocks.{idx}/attn/qkv/Add"], 0, 1)
    oh.connect_nodes(model, [new_concat_nodes[0], f"/blocks.{idx}/attn/Mul"], 0, 0)
    oh.connect_nodes(model, [f"/blocks.{idx}/attn/Mul", new_einsum_nodes[0]], 0, 0)
    oh.connect_nodes(model, [new_concat_nodes[1], new_einsum_nodes[0]], 0, 1)
    oh.connect_nodes(model, [new_einsum_nodes[0], f"/blocks.{idx}/attn/Softmax"], 0, 0)
    oh.connect_nodes(model, [f"/blocks.{idx}/attn/Softmax", new_einsum_nodes[1]], 0, 0)
    oh.connect_nodes(model, [new_concat_nodes[2], new_einsum_nodes[1]], 0, 1)
    oh.connect_nodes(model, [new_concat_nodes[3], new_conv_nodes[1]], 0, 0)
    oh.connect_nodes(model, [new_conv_nodes[1], f"/blocks.{idx}/attn/proj/Add"], 0, 1)


def _modify_mlp_post(idx, model, transformer_blocks_mlp, start):
    at_node = f"/blocks.{idx}/norm2/LayerNormalization"
    perm_before = (0, 2, 3, 1)
    perm_after = (0, 3, 1, 2)
    new_transpose_nodes = oh.insert_transpose_pair(model, at_node, perm_before, perm_after)

    at_nodes = [
        f"/blocks.{idx}/mlp/fc1/MatMul",
        f"/blocks.{idx}/mlp/fc2/MatMul"
    ]
    new_conv_nodes = oh.rewrite_matmul_as_conv(model, at_nodes)

    # Update constants
    node = oh.find_node(model, f"/blocks.{idx}/mlp/fc1/Add")
    param_name = node.input[0]  # "blocks.{idx}.mlp.fc1.bias"
    perm = None
    new_shape = (1, 1536, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)
    node = oh.find_node(model, f"/blocks.{idx}/mlp/fc2/Add")
    param_name = node.input[0]  # "blocks.{idx}.mlp.fc2.bias"
    perm = None
    new_shape = (1, 384, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)
    node = oh.find_node(model, f"/blocks.{idx}/ls2/Mul")
    param_name = node.input[1]  # "blocks.{idx}.ls2.gamma"
    perm = None
    new_shape = (1, 384, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)

    # Connect newly created nodes
    oh.connect_nodes(model, [new_transpose_nodes[1], new_conv_nodes[0]], 0, 0)
    oh.connect_nodes(model, [new_conv_nodes[0], f"/blocks.{idx}/mlp/fc1/Add"], 0, 1)
    oh.connect_nodes(model, [new_conv_nodes[1], f"/blocks.{idx}/mlp/fc2/Add"], 0, 1)


def _modify_postproc(k: int):
    postproc_fname = _postproc_split.filename
    model = onnx.load(postproc_fname)
    graph_def = model.graph
    initializers = graph_def.initializer

    # Define the start and end blocks
    start_block = 3
    end_block = 12

    # Modify attn and mlp for the specified blocks
    transformer_blocks_attn, transformer_blocks_mlp = _split_transformer_post(start_block, end_block)
    for block in range(start_block, end_block):
        _modify_attn_post(block, model, transformer_blocks_attn, start_block)
        _modify_mlp_post(block, model, transformer_blocks_mlp, start_block)


    # Update model input: (1, 257, 384) -> (1, 384, 1, 257)
    oh.update_io_shape(model, _postproc_split.input_names[0], (1, 384, 1, 1370))

    # Insert transposes
    transpose_nodes = []
    skip_nodes = []

    # add transposes for final LayerNorms
    for suffix in ['', '_1', '_2', '_3']:
        at_node = "/norm" + suffix + "/LayerNormalization"
        skip_nodes.append(at_node)
        perm_before = (0, 2, 3, 1)
        perm_after = (0, 3, 1, 2)
        new_transpose_nodes = oh.insert_transpose_pair(model, at_node, perm_before, perm_after)
        transpose_nodes.append(new_transpose_nodes)

        for node in model.graph.node:
            if node.name == "/Slice" + suffix:
                new_initializer_name = f"slice_axes{suffix}"
                initializers.append(
                    onnx.helper.make_tensor(new_initializer_name, onnx.TensorProto.INT64, [1], [2])
                )

                node.input[3] = new_initializer_name

    i = 0
    for node_idx, node in enumerate(graph_def.node):
        if "/depth_head/Transpose" in node.name:
            model.graph.node.remove(node)

            # Insert a slice node per row
            slice_output = f"/Slice_{i}_output_0" if i > 0 else "/Slice_output_0"
            slice_axes = f"/depth_head/slice_axes{i}"
            initializers.append(
                onnx.helper.make_tensor(slice_axes, onnx.TensorProto.INT64, [1], [2])
            )
            split_outputs = []
            for h in range(37):
                slice_starts = f"/depth_head/slice_{i}_{h}_starts"
                slice_ends = f"/depth_head/slice_{i}_{h}_ends"
                initializers.append(
                    onnx.helper.make_tensor(slice_starts, onnx.TensorProto.INT64, [1], [h * 37])
                )
                initializers.append(
                    onnx.helper.make_tensor(slice_ends, onnx.TensorProto.INT64, [1], [(h + 1) * 37])
                )
                output = f"/depth_head/Slice_{i}_{h}"
                slice_node = onnx.helper.make_node(
                    op_type="Slice",
                    name=f"/depth_head/Slice_{i}_{h}",
                    inputs=[slice_output, slice_starts, slice_ends, slice_axes],
                    outputs=[output])
                model.graph.node.insert(node_idx + h, slice_node)
                split_outputs.append(output)
            i += 1

        elif "/depth_head/Reshape" in node.name:
            name = node.name + "/Concat"
            transpose_name = name[1:] + "/Transpose"
            outputs = node.output
            model.graph.node.remove(node)


            # Insert a concat node
            model.graph.node.insert(
                node_idx,
                onnx.helper.make_node(
                    op_type="Concat",
                    name=name,
                    inputs=split_outputs,
                    outputs=[name + '_output'],
                    axis=1
                )
            )

            model.graph.node.insert(
                node_idx + 1,
                onnx.helper.make_node(
                    op_type="Transpose",
                    name=transpose_name,
                    inputs=[name + '_output'],
                    outputs=outputs,
                    perm=(0, 3, 1, 2)
                ))



    for value_info in list(model.graph.value_info):
        model.graph.value_info.remove(value_info)


    for initializer in model.graph.initializer:
        if initializer.name == "/depth_head/refinenet4/Concat_output_0":
            # Update the tensor values to [1, 128, 34, 34]
            new_values = np.array([1, 64, 38, 38], dtype=np.int64)
            new_tensor = numpy_helper.from_array(new_values, initializer.name)

            # Replace the old initializer with the updated one
            model.graph.initializer.remove(initializer)
            model.graph.initializer.append(new_tensor)
            break


    # Update first Resize
    initializers = model.graph.initializer

    slice_axes_2 = "slice_axes_resize_2"
    initializers.append(
        onnx.helper.make_tensor(slice_axes_2, onnx.TensorProto.INT64, [1], [2])
    )

    slice_axes_3 = "slice_axes_resize_3"
    initializers.append(
        onnx.helper.make_tensor(slice_axes_3, onnx.TensorProto.INT64, [1], [3])
    )

    slice_starts_resize = "slice_starts_resize"
    initializers.append(
        onnx.helper.make_tensor(slice_starts_resize, onnx.TensorProto.INT64, [1], [1])
    )

    slice_step_resize = "slice_step_resize"
    initializers.append(
        onnx.helper.make_tensor(slice_step_resize, onnx.TensorProto.INT64, [1], [1])
    )

    slice_ends_int = "slice_ends_int"
    initializers.append(
        onnx.helper.make_tensor(slice_ends_int, onnx.TensorProto.INT64, [1], [38])
    )

    for node_idx, node in enumerate(model.graph.node):
        if node.name == "/depth_head/refinenet4/Resize":
            slice_node_1 = onnx.helper.make_node(
                op_type="Slice",
                name="/depth_head_refinenet4/Slice_1",
                inputs=["/depth_head/refinenet4/Resize_output_0", slice_starts_resize,
                        slice_ends_int, slice_axes_3, slice_step_resize],
                outputs=["/depth_head_refinenet4/Resize/Slice_1_output"])
            model.graph.node.insert(node_idx + 1, slice_node_1)

            slice_node_2 = onnx.helper.make_node(
                op_type="Slice",
                name="/depth_head_refinenet4/Slice_2",
                inputs=["/depth_head_refinenet4/Resize/Slice_1_output", slice_starts_resize,
                        slice_ends_int, slice_axes_2, slice_step_resize],
                outputs=["/depth_head_refinenet4/Resize/Slice_2_output"])
            model.graph.node.insert(node_idx + 2, slice_node_2)

    for i, node in enumerate(model.graph.node):
        if node.name == "/depth_head/refinenet4/out_conv/Conv":
            node.input[0] = "/depth_head_refinenet4/Resize/Slice_2_output"


    # Update second Resize
    for initializer in model.graph.initializer:
        if initializer.name == "/depth_head/Concat_output_0":
            # Update the tensor values to [1, 128, 34, 34]
            new_values = np.array([1, 32, 592, 592], dtype=np.int64)
            new_tensor = numpy_helper.from_array(new_values, initializer.name)

            # Replace the old initializer with the updated one
            model.graph.initializer.remove(initializer)
            model.graph.initializer.append(new_tensor)
            break


    # Find the node index where the Resize operation is located
    for node_idx, node in enumerate(model.graph.node):
        if node.name == "/depth_head/Resize":
            break

    total_size = 592
    target_size = _input_shape[2]

    step = total_size / target_size

    # Generate the indices that will be kept to reduce the size from 592 to 518
    # Generate the indices that need to be removed to achieve the target size
    all_indices = np.arange(total_size)
    keep_indices = np.arange(0, total_size, step).astype(np.int64)
    keep_indices = np.round(keep_indices).astype(np.int64)

    rmv_indices = np.setdiff1d(all_indices, keep_indices)
    rmv_indices = rmv_indices[k-1::k]

    result = np.concatenate([rmv_index + np.arange(k) for rmv_index in rmv_indices])

    keep_indices = np.setdiff1d(all_indices, result)


    # Ensure there are exactly target_size elements
    if len(keep_indices) > target_size:
        keep_indices = keep_indices[:target_size]
    elif len(keep_indices) < target_size:
        keep_indices = np.append(keep_indices, total_size - 1)


    # Find the missing indices
    height_indices = np.setdiff1d(all_indices, keep_indices)
    width_indices = height_indices  # Same for height and width

    # Find the node index where the Resize operation is located
    for node_idx, node in enumerate(model.graph.node):
        if node.name == "/depth_head/Resize":
            break

    # Create and insert the Slice nodes for height dimension
    height_slices = []
    for i, index in enumerate(height_indices[::k]):
        height_start_tmp = height_indices[k*i - 1] + 1 if i != 0 else 0
        start_height = np.array([0, 0, height_start_tmp, 0], dtype=np.int64)

        if i == 0:
            height_end_tmp = height_indices[i]
        else:
            height_end_tmp = height_indices[k*i+1-1] if k*i < len(height_indices) - (k-1) else 592
        end_height = np.array([1, 32, height_end_tmp, 592], dtype=np.int64)
        axes = np.array([0, 1, 2, 3], dtype=np.int64)

        start_height_tensor = numpy_helper.from_array(start_height, name=f'start_height_{i}')
        end_height_tensor = numpy_helper.from_array(end_height, name=f'end_height_{i}')
        axes_tensor = numpy_helper.from_array(axes, name=f'axes_{i}')

        model.graph.initializer.append(start_height_tensor)
        model.graph.initializer.append(end_height_tensor)
        model.graph.initializer.append(axes_tensor)

        slice_height_node = helper.make_node(
            'Slice',
            inputs=[node.output[0], f'start_height_{i}', f'end_height_{i}', f'axes_{i}'],
            outputs=[f'sliced_height_{i}'],
            name=f'slice_height_{i}'
        )
        model.graph.node.insert(node_idx + 1 + i, slice_height_node)
        height_slices.append(f'sliced_height_{i}')

    # Create the Concat node to merge height slices
    concat_height_node = helper.make_node(
        'Concat',
        inputs=height_slices,
        outputs=['concatenated_height'],
        axis=2  # Assuming height slices are concatenated along height dimension
    )
    model.graph.node.insert(node_idx + 1 + int(np.ceil(len(height_indices) / k)), concat_height_node)

    # Create and insert the Slice nodes for width dimension
    width_slices = []
    for i, index in enumerate(width_indices[::k]):
        width_start_tmp = width_indices[k*i - 1] + 1 if i != 0 else 0
        start_width = np.array([0, 0, 0, width_start_tmp], dtype=np.int64)
        # width_end_tmp = width_indices[i] if i < len(width_indices) else 592

        if i == 0:
            width_end_tmp = width_indices[i]
        else:
            width_end_tmp = width_indices[k*i+1-1] if k*i < len(width_indices) - (k-1) else 592

        end_width = np.array([1, 32, _input_shape[2], width_end_tmp], dtype=np.int64)
        axes = np.array([0, 1, 2, 3], dtype=np.int64)

        start_width_tensor = numpy_helper.from_array(start_width, name=f'start_width_{i}')
        end_width_tensor = numpy_helper.from_array(end_width, name=f'end_width_{i}')
        axes_tensor = numpy_helper.from_array(axes, name=f'axes_width_{i}')

        model.graph.initializer.append(start_width_tensor)
        model.graph.initializer.append(end_width_tensor)
        model.graph.initializer.append(axes_tensor)

        slice_width_node = helper.make_node(
            'Slice',
            inputs=['concatenated_height', f'start_width_{i}', f'end_width_{i}', f'axes_width_{i}'],
            outputs=[f'sliced_width_{i}'],
            name=f'slice_width_{i}'
        )
        model.graph.node.insert(node_idx + 2 + int(np.ceil(len(height_indices) / k)) + i, slice_width_node)
        width_slices.append(f'sliced_width_{i}')

    # Create the final Concat node to merge width slices
    concat_width_node = helper.make_node(
        'Concat',
        inputs=width_slices,
        outputs=['concatenated_width'],
        axis=3  # Assuming width slices are concatenated along width dimension
    )
    model.graph.node.insert(node_idx + 2 + int(np.ceil(len(height_indices) / k)) + int(np.ceil(len(width_indices) / k)), concat_width_node)



    for i, node in enumerate(model.graph.node):
        if node.name == "/depth_head/output_conv2/output_conv2.0/Conv":
            node.input[0] = "concatenated_width"
            break

    if val:
        # Update model output
        # for i, node in enumerate(model.graph.node):
        #     if node.name == "/Mul_1":
        #         node.output[0] = "output"
        #     if node.name == "/Squeeze":
        #         model.graph.node.remove(node)
        for i, node in enumerate(model.graph.node):
            if node.name == "/Mul_1":
                relu_node = node
                node_idx = i
                # node.output[0] = "output"
            if node.name == "/Squeeze":
                model.graph.node.remove(node)

        model.graph.node.insert(
            node_idx + 1,
            onnx.helper.make_node(
                op_type="Transpose",
                name="Transpose",
                inputs=relu_node.output,
                outputs=["transposed_output"],
                perm=(0, 2, 3, 1)
            )
        )

    else:
        # Update model output
        # for i, node in enumerate(model.graph.node):
        #     if node.name == "/Relu":
        #         node.output[0] = "output"
        #     if node.name == "/Squeeze":
        #         model.graph.node.remove(node)
        for i, node in enumerate(model.graph.node):
            if node.name == "/Relu":
                relu_node = node
                node_idx = i
                # node.output[0] = "output"
            if node.name == "/Squeeze":
                model.graph.node.remove(node)

        model.graph.node.insert(
            node_idx + 1,
            onnx.helper.make_node(
                op_type="Transpose",
                name="Transpose",
                inputs=relu_node.output,
                outputs=["transposed_output"],
                perm=(0, 2, 3, 1)
            )
        )

    output_shape = [1, 518, 518, 1]
    output_type = onnx.TensorProto.FLOAT
    output_name = "transposed_output"

    output_proto = onnx.helper.make_tensor_value_info(output_name, output_type, output_shape)
    model.graph.output.pop(0)
    model.graph.output.append(output_proto)


    model, check = simplify(model)
    assert check, "Opt for preproc: Simplified ONNX model can not be validated"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_onnx_fname = Path(postproc_fname.stem + "_opt.onnx")
    onnx.save(model, output_onnx_fname)

    print(f'Opt for preproc ONNX file saved to: {output_onnx_fname}')
    return output_onnx_fname


def _verify_opt_postproc(src_model, opt_model, input_names):
    # Input: (1, 257, 384) -> (1, 384, 1, 257)
    # Output: (1, 384) -> (1, 384, 1, 1)
    orig_input = _generate_test_vector((1, 1370, 384))
    opt_input = orig_input.transpose((0, 2, 1)).reshape(1, 384, 1, 1370)

    ref_outputs = oh.run_model(src_model, input_names, [orig_input])
    new_outputs = oh.run_model(opt_model, input_names, [opt_input])

    diff = np.transpose(ref_outputs[0], (1, 2, 0)) - new_outputs[0]
    print(f"Error: min = {np.min(diff)}, max = {np.max(diff)}")


_ops_to_be_removed: Dict[str, Dict[str, int]] = {
    "attn": {
        "Reshape": 2,
        "Transpose": 3,
        "Gather": 3
    }
}


def _make_remove_list(blk: int, part: str, ops_dict: Dict[str, int]):
    remove_list = []
    for k, v in ops_dict.items():
        for i in range(v):
            node_name = f"/blocks{blk}/{part}/{k}" if i == 0 else f"/blocks{blk}/{part}/{k}_{i}"
            remove_list.append(node_name)
    return remove_list


def _modify_attn(idx):
    global transformer_blocks_attn
    model_fname = transformer_blocks_attn[idx].filename
    model = onnx.load(model_fname)

    # Remove nodes
    part = "attn"
    remove_list = _make_remove_list(idx, part, _ops_to_be_removed[part])
    oh.remove_nodes_by_name_list(model, remove_list)

    # Insert transpose pair
    at_node = f"/blocks.{idx}/norm1/LayerNormalization"
    perm_before = (0, 2, 3, 1)
    perm_after = (0, 3, 1, 2)
    new_transpose_nodes = oh.insert_transpose_pair(model, at_node, perm_before, perm_after)

    # 2x Replace MatMul with Conv
    at_nodes = [
        f"/blocks.{idx}/attn/qkv/MatMul",
        f"/blocks.{idx}/attn/proj/MatMul"
    ]
    new_conv_nodes = oh.rewrite_matmul_as_conv(model, at_nodes)

    # 2x Replace MatMul with Einsum
    equations = {
        f"/blocks.{idx}/attn/MatMul": "nchw,nchq->nqhw",
        f"/blocks.{idx}/attn/MatMul_1": "nchw,nqhc->nqhw"
    }
    new_einsum_nodes = oh.rewrite_matmul_as_einsum(model, equations)

    # Insert slices
    after_node = f"/blocks.{idx}/attn/qkv/Add"
    slices_after_add = oh.insert_slices_after_node(model, after_node, axis=1, nslices=3, slice_size=384)

    # 3x Insert slices-concat
    nodes_after_concat = [
        f"/blocks.{idx}/attn/Mul",
        new_einsum_nodes[0],
        new_einsum_nodes[1]
    ]
    new_concat_nodes = []
    for i in range(3):
        new_node = oh.insert_slices_concat_between_nodes(model, slices_after_add[i], nodes_after_concat[i],
                                                         slice_axis=1, nslices=6, slice_size=64,
                                                         concat_axis=2)
        new_concat_nodes.append(new_node)

    # 1x Insert slices-concat
    after_node = new_einsum_nodes[1]
    before_node = new_conv_nodes[1]
    new_node = oh.insert_slices_concat_between_nodes(model, after_node, before_node,
                                                     slice_axis=2, nslices=6, slice_size=1,
                                                     concat_axis=1)
    new_concat_nodes.append(new_node)

    # Update I/O shape: (1, 257, 384) -> (1, 384, 1, 257)
    oh.update_io_shape(model, transformer_blocks_attn[idx].input_names[0], (1, 384, 1, 1370))
    oh.update_io_shape(model, transformer_blocks_attn[idx].output_names[0], (1, 384, 1, 1370))

    # Update constants
    node = oh.find_node(model, f"/blocks.{idx}/attn/qkv/Add")
    param_name = node.input[0]  # "blocks.0.attn.qkv.bias"
    perm = None
    new_shape = (1, 1152, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)
    node = oh.find_node(model, f"/blocks.{idx}/attn/proj/Add")
    param_name = node.input[0]  # "blocks.0.attn.proj.bias"
    perm = None
    new_shape = (1, 384, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)
    node = oh.find_node(model, f"/blocks.{idx}/ls1/Mul")
    param_name = node.input[1]  # "blocks.0.ls1.gamma"
    perm = None
    new_shape = (1, 384, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)

    # Update attributes
    oh.set_attribute_to_node(model, f"/blocks.{idx}/attn/Softmax", "axis", 1)

    # Connect newly created nodes
    oh.connect_nodes(model, [new_transpose_nodes[1], new_conv_nodes[0]], 0, 0)
    oh.connect_nodes(model, [new_conv_nodes[0], f"/blocks.{idx}/attn/qkv/Add"], 0, 1)
    oh.connect_nodes(model, [new_concat_nodes[0], f"/blocks.{idx}/attn/Mul"], 0, 0)
    oh.connect_nodes(model, [f"/blocks.{idx}/attn/Mul", new_einsum_nodes[0]], 0, 0)
    oh.connect_nodes(model, [new_concat_nodes[1], new_einsum_nodes[0]], 0, 1)
    oh.connect_nodes(model, [new_einsum_nodes[0], f"/blocks.{idx}/attn/Softmax"], 0, 0)
    oh.connect_nodes(model, [f"/blocks.{idx}/attn/Softmax", new_einsum_nodes[1]], 0, 0)
    oh.connect_nodes(model, [new_concat_nodes[2], new_einsum_nodes[1]], 0, 1)
    oh.connect_nodes(model, [new_concat_nodes[3], new_conv_nodes[1]], 0, 0)
    oh.connect_nodes(model, [new_conv_nodes[1], f"/blocks.{idx}/attn/proj/Add"], 0, 1)

    # Remove existing shape inference results.
    for value_info in list(model.graph.value_info):
        model.graph.value_info.remove(value_info)

    model, check = simplify(model)
    assert check, f"Opt for attn_{idx}: Simplified ONNX model can not be validated"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_onnx_fname = Path(model_fname.stem + "_opt.onnx")
    onnx.save(model, output_onnx_fname)

    print(f'Opt for attn ONNX file saved to: {output_onnx_fname}')
    return output_onnx_fname


def _modify_mlp(idx):
    global transformer_blocks_mlp
    model_fname = transformer_blocks_mlp[idx].filename
    model = onnx.load(model_fname)

    # Insert transpose pair
    at_node = f"/blocks.{idx}/norm2/LayerNormalization"
    perm_before = (0, 2, 3, 1)
    perm_after = (0, 3, 1, 2)
    new_transpose_nodes = oh.insert_transpose_pair(model, at_node, perm_before, perm_after)

    at_nodes = [
        f"/blocks.{idx}/mlp/fc1/MatMul",
        f"/blocks.{idx}/mlp/fc2/MatMul"
    ]
    new_conv_nodes = oh.rewrite_matmul_as_conv(model, at_nodes)

    # Update I/O shape: (1, 257, 384) -> (1, 384, 1, 257)
    oh.update_io_shape(model, transformer_blocks_mlp[idx].input_names[0], (1, 384, 1, 1370))
    oh.update_io_shape(model, transformer_blocks_mlp[idx].output_names[0], (1, 384, 1, 1370))

    # Update constants
    node = oh.find_node(model, f"/blocks.{idx}/mlp/fc1/Add")
    param_name = node.input[0]  # "blocks.{idx}.mlp.fc1.bias"
    perm = None
    new_shape = (1, 1536, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)
    node = oh.find_node(model, f"/blocks.{idx}/mlp/fc2/Add")
    param_name = node.input[0]  # "blocks.{idx}.mlp.fc2.bias"
    perm = None
    new_shape = (1, 384, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)
    node = oh.find_node(model, f"/blocks.{idx}/ls2/Mul")
    param_name = node.input[1]  # "blocks.{idx}.ls2.gamma"
    perm = None
    new_shape = (1, 384, 1, 1)
    oh.transpose_reshape_constant(model, param_name, perm, new_shape)

    # Connect newly created nodes
    oh.connect_nodes(model, [new_transpose_nodes[1], new_conv_nodes[0]], 0, 0)
    oh.connect_nodes(model, [new_conv_nodes[0], f"/blocks.{idx}/mlp/fc1/Add"], 0, 1)
    oh.connect_nodes(model, [new_conv_nodes[1], f"/blocks.{idx}/mlp/fc2/Add"], 0, 1)

    # Remove existing shape inference results.
    for value_info in list(model.graph.value_info):
        model.graph.value_info.remove(value_info)

    model, check = simplify(model)
    assert check, f"Opt for mlp_{idx}: Simplified ONNX model can not be validated"
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    output_onnx_fname = Path(model_fname.stem + "_opt.onnx")
    onnx.save(model, output_onnx_fname)

    print(f'Opt for mlp ONNX file saved to: {output_onnx_fname}')
    return output_onnx_fname


def _verify_opt_block(src_model, opt_model, input_names):
    # Input: (1, 257, 384) -> (1, 384, 1, 257)
    # Output: (1, 257, 384) -> (1, 384, 1, 257)
    orig_input = _generate_test_vector((1, 1370, 384))
    opt_input = orig_input.transpose((0, 2, 1)).reshape(1, 384, 1, 1370)

    ref_outputs = oh.run_model(src_model, input_names, [orig_input])
    new_outputs = oh.run_model(opt_model, input_names, [opt_input])

    opt_outputs = []
    for no in new_outputs:
        t = no.reshape(1, 384, 1370)
        t = t.transpose((0, 2, 1))
        opt_outputs.append(t)

    diff = ref_outputs[0] - opt_outputs[0]
    print(f"Error: min = {np.min(diff)}, max = {np.max(diff)}")
    # assert np.allclose(ref_outputs[0], opt_outputs[0])
    # print(f"Verification OK - opt model {opt_model}")


def _verify_opt_whole_model(src_model, opt_model, input_names):
    same_input = _generate_test_vector((1, 3, _input_shape[2], _input_shape[2]))

    ref_outputs = oh.run_model(src_model, input_names, [same_input])
    new_outputs = oh.run_model(opt_model, input_names, [same_input])

    diff = np.transpose(ref_outputs[0], (1, 2, 0)) - new_outputs[0]
    print(f"Error: min = {np.min(diff)}, max = {np.max(diff)}")


def main(*, verify_simplified, split_top_level, split_transformer, verify_all_splits, gen_opt, num_splits=2, k):
    global transformer_blocks_attn, transformer_blocks_mlp
    input_data = _generate_test_vector(_input_shape)

    # --------- simplified model ------------------

    # Verify simplified model
    if verify_simplified:
        oh.verify_models_equal(_source_onnx_fname, _simplified_onnx_fname, [_model_input_name], [input_data])

    # --------- model splits ------------------

    # Verify top level split
    if split_top_level:
        oh.split_model(_top_level_splits)
        oh.verify_split_models(_simplified_onnx_fname, _top_level_splits, [_model_input_name], [input_data])

    # Generate List[_ModelSplit] for transformer blocks
    transformer_blocks_attn, transformer_blocks_mlp = _split_transformer(_MIDLE_BLOCKS)

    # Split ViT transformer
    if split_transformer:
        oh.split_model(transformer_blocks_attn)
        oh.split_model(transformer_blocks_mlp)

    # Verify all splits
    if verify_all_splits:
        transformer_blocks = [v for pair in zip(transformer_blocks_attn, transformer_blocks_mlp) for v in pair]
        all_splits = [_preproc_split] + transformer_blocks + [_postproc_split]
        oh.verify_split_models(_simplified_onnx_fname, all_splits, [_model_input_name], [input_data])

    # --------- model rewrite ------------------

    if gen_opt:

        opt_preproc_fname = _modify_preproc()
        _verify_opt_preproc(_preproc_split.filename, opt_preproc_fname, [_model_input_name], [input_data])

        opt_postproc_fname = _modify_postproc(k)
        _verify_opt_postproc(_postproc_split.filename, opt_postproc_fname, _postproc_split.input_names)

        opt_attn_fname = []
        opt_mlp_fname = []
        for idx in range(_MIDLE_BLOCKS):
            opt_fname = _modify_attn(idx)
            opt_attn_fname.append(opt_fname)
            _verify_opt_block(transformer_blocks_attn[idx].filename, opt_fname,
                              transformer_blocks_attn[idx].input_names)

            opt_fname = _modify_mlp(idx)
            opt_mlp_fname.append(opt_fname)
            _verify_opt_block(transformer_blocks_mlp[idx].filename, opt_fname, transformer_blocks_mlp[idx].input_names)

        opt_model_fname = _simplified_onnx_fname.stem + "_opt.onnx"

        new_model = oh.merge_split_model_with_shared_constant(None, opt_preproc_fname)
        for idx in range(_MIDLE_BLOCKS):
            new_model = oh.merge_split_model_with_shared_constant(new_model, opt_attn_fname[idx], f":attn{idx}")
            new_model = oh.merge_split_model_with_shared_constant(new_model, opt_mlp_fname[idx], f":mlp{idx}")
        new_model = oh.merge_split_model_with_shared_constant(new_model, opt_postproc_fname)

        model_opt, check = simplify(new_model)
        assert check, "Opt - Simplified ONNX model can not be validated"
        onnx.checker.check_model(model_opt)
        onnx.save(model_opt, opt_model_fname)
        print(f"Opt/Modified ONNX file saved to: {opt_model_fname}")

        _verify_opt_whole_model(_simplified_onnx_fname, opt_model_fname, [_model_input_name])

        model = model_opt
        nodes = list(model.graph.node)

        # pass #1
        for idx, node in enumerate(nodes):
            if (len(nodes) > idx + 3 and node.op_type == 'Einsum' and
                    nodes[idx + 1].op_type == 'Softmax' and
                    nodes[idx + 2].op_type == 'Concat' and
                    nodes[idx + 3].op_type == 'Einsum'):
                print('--pattern at', idx)

                split_einsum_1 = onnx.helper.make_node(
                    name=node.name + "_Split", op_type="Split",
                    inputs=[node.input[0]],
                    outputs=[node.name + "_split_out_" + str(i) for i in range(num_splits)],
                    axis=3,
                )
                # einsum at i
                einsums = make_einsums(split_einsum_1.output, node, node.attribute[0].s, num=num_splits)
                einsum_output = [e.output[0] for e in einsums]
                _replace_node(model, node, [split_einsum_1, *einsums])

                # softmax at i+1
                softmaxs = make_softmaxs(nodes[idx + 1], einsum_output, num=num_splits)
                _replace_node(model, nodes[idx + 1], softmaxs)

                # einsum at i+3
                node_i3 = nodes[idx + 3]
                softmax_outs = [s.output[0] for s in softmaxs]
                einsums_i3 = make_einsums(softmax_outs, node_i3, node_i3.attribute[0].s, num=num_splits)

                concat = onnx.helper.make_node(
                    name=node_i3.name + "__concat", op_type="Concat",
                    inputs=[e.output[0] for e in einsums_i3],
                    outputs=[node_i3.output[0]], axis=3
                )
                _replace_node(model, node_i3, [*einsums_i3, concat])

        final_model_name = Path(args.model[:-5] + f"_opt.onnx")
        oh.save_model(model, final_model_name)
        print(f'ONNX file saved to {final_model_name}')

        # _verify(final_model_name, _onnx_fname)
        # _verify(final_model_name, Path(f"{_onnx_path[:-5]}_opt.onnx"))
        _verify(final_model_name, opt_model_fname)

    print(f'Final model name: {final_model_name}')

    stemed_model_name = os.path.basename(_onnx_path.stem)

    os.remove(f"{stemed_model_name}_simplified_opt.onnx")
    os.remove(f"{stemed_model_name}_simplified_postproc.onnx")
    os.remove(f"{stemed_model_name}_simplified_postproc_opt.onnx")
    os.remove(f"{stemed_model_name}_simplified_preproc.onnx")
    os.remove(f"{stemed_model_name}_simplified_preproc_opt.onnx")
    os.remove(f"{stemed_model_name}_simplified_vit.onnx")

    for i in range(_MIDLE_BLOCKS):
        os.remove(f"{stemed_model_name}_simplified_vit_block.{i}_attn.onnx")
        os.remove(f"{stemed_model_name}_simplified_vit_block.{i}_attn_opt.onnx")
        os.remove(f"{stemed_model_name}_simplified_vit_block.{i}_mlp.onnx")
        os.remove(f"{stemed_model_name}_simplified_vit_block.{i}_mlp_opt.onnx")

    print("--Done!")


def _verify(model_name, ref_model_name):
    # img = _generate_test_vector_no_shape()
    # assert img.shape == _input_shape, \
    #     f"Illegal test vector shape. Got {img.shape}, expected {_input_shape}"

    img = np.random.uniform(low=-1.0, high=1.0, size=_input_shape).astype(np.float32)

    ref_outputs = _run_model(ref_model_name, ["input"], [img])

    print('verify.ref outs.len:', len(ref_outputs), '\n')

    outputs = _run_model(model_name, ["input"], [img])

    print('verify.ref outs.len:', len(ref_outputs), '\n',
          ' mod outs.len:', len(outputs))

    assert len(ref_outputs) == 1

    assert np.array_equal(ref_outputs[0], outputs[0]) # for i in range(len(outputs))

    print(f'ONNX file {model_name} verified!\n')

if __name__ == "__main__":
    main(
        verify_simplified=True,
        split_top_level=True,
        split_transformer=True,
        verify_all_splits=True,
        gen_opt=True,
        k=args.k
    )