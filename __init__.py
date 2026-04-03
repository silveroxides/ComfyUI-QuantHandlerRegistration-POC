"""
POC: Custom node __init__.py that registers INT8 quant layouts via QuantHandler API.

This is the minimal pattern for any custom node that wants to add quantization
formats to ComfyUI's loader pipeline using the register_quant_handler PR.
"""
import logging
import torch

# ── 1. Import the registration API from ComfyUI core ────────────────────────
from comfy.quant_ops import QuantHandler, register_quant_handler

# ── 2. Import layout classes from comfy_kitchen ─────────────────────────────
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout, BlockWiseINT8Layout


# ── 3. Define build_params callables ────────────────────────────────────────
#
# These tell ops.py how to construct layout Params from a safetensors state
# dict during model loading. Signature is fixed by the QuantHandler contract:
#
#   (layout_cls, state_dict, prefix, device,
#    manually_loaded_keys, compute_dtype,
#    out_features, in_features, load_scale_fn) -> layout_cls.Params


def build_int8_tensorwise_params(
    layout_cls, state_dict, prefix, device,
    manually_loaded_keys, compute_dtype,
    out_features, in_features, load_scale_fn,
):
    """Build Params for TensorWiseINT8Layout — single scale per tensor."""
    scale = load_scale_fn(state_dict, prefix, "weight_scale", device, manually_loaded_keys)
    return layout_cls.Params(
        scale=scale,
        orig_dtype=compute_dtype,
        orig_shape=(out_features, in_features),
        is_weight=True,
    )


def build_int8_blockwise_params(
    layout_cls, state_dict, prefix, device,
    manually_loaded_keys, compute_dtype,
    out_features, in_features, load_scale_fn,
):
    """Build Params for BlockWiseINT8Layout — per-block scale, block_size=128."""
    scale = load_scale_fn(state_dict, prefix, "weight_scale", device, manually_loaded_keys)
    return layout_cls.Params(
        scale=scale,
        orig_dtype=compute_dtype,
        orig_shape=(out_features, in_features),
        block_size=128,
        is_weight=True,
    )


# ── 4. Register handlers ───────────────────────────────────────────────────
#
# register_quant_handler does three things per call:
#   a) Registers the layout class with comfy_kitchen (register_layout_class)
#   b) Adds a QuantHandler entry to QUANT_HANDLERS (used by ops.py fallback)
#   c) Adds a backward-compat entry to QUANT_ALGOS + QUANT_ALGO_NAMES

register_quant_handler("int8_tensorwise", QuantHandler(
    layout_class=TensorWiseINT8Layout,
    storage_t=torch.int8,
    parameters={"weight_scale", "input_scale"},
    comfy_tensor_layout="TensorWiseINT8Layout",
    build_params=build_int8_tensorwise_params,
))

register_quant_handler("int8_blockwise", QuantHandler(
    layout_class=BlockWiseINT8Layout,
    storage_t=torch.int8,
    parameters={"weight_scale", "input_scale"},
    comfy_tensor_layout="BlockWiseINT8Layout",
    group_size=128,
    build_params=build_int8_blockwise_params,
))

logging.info("POC: Registered int8_tensorwise and int8_blockwise via QuantHandler")
