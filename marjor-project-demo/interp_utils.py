import torch
from torch import Tensor
import torch.nn.functional as F
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache
from einops import einsum

from jaxtyping import Int, Float
from typing import List, Tuple, Optional, Literal
import numpy as np
from transformer_lens import utils

# ---------------------------------------------------------------------------------------------------------------- #

def get_layer_attributions(
    model: HookedTransformer,
    typ: Literal["attn", "mlp"],
    cache: dict[str, Tensor],
    target_idx: Int
) -> Tensor:
    
    target_U = model.W_U[:, target_idx]

    attrib_list = []
    for layer in range(model.cfg.n_layers):
        block_out = cache[f"blocks.{layer}.hook_{typ}_out"][-1]
        attrib_list.append(block_out @ target_U)

    return torch.tensor(attrib_list)

# ---------------------------------------------------------------------------------------------------------------- #

def decomposed_head_attribs(
    model: HookedTransformer,
    cache: dict[str, Tensor],
    target_idx: Int,
) -> Float[Tensor, "n_layers n_heads"]:
    
    ret_attribs = torch.zeros(12, 12)
    for layer in range(model.cfg.n_layers):    
        decomp_head_out = einsum(
                            cache[f"blocks.{layer}.attn.hook_z"],
                            model.blocks[layer].attn.W_O,
                            "seq n_heads d_head, n_heads d_head d_model -> seq n_heads d_model"
                        )
    
        dec_head_attribs = einsum(
                            decomp_head_out[-1],
                            model.W_U[:, target_idx],
                            "n_heads d_model, d_model -> n_heads",
                        )
        ret_attribs[layer] = dec_head_attribs

    return ret_attribs

# ---------------------------------------------------------------------------------------------------------------- #

def get_top_percent_activations(
    model: HookedTransformer,
    cache: ActivationCache,
    psent: int=1,
) -> List:

    layer_tops = []

    for layer in range(model.cfg.n_layers):
        layer_act = cache[f"blocks.{layer}.mlp.hook_post"][-1]
        k = max(1, int(len(layer_act) * psent / 100))
        
        top_vals, top_indices = torch.topk(layer_act, k=k)
        layer_tops.append((top_vals, top_indices))

    return layer_tops
