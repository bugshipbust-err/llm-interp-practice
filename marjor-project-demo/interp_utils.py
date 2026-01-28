import torch
from torch import Tensor
import torch.nn.functional as F
from einops import einsum

import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache

import sae_lens
from sae_lens import SAE, ActivationsStore, HookedSAETransformer, LanguageModelSAERunnerConfig
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from tqdm.auto import tqdm
from jaxtyping import Int, Float
from typing import List, Tuple, Optional, Literal
import numpy as np
from transformer_lens import utils

# ---------------------------------------------------------------------------------------------------------------- #

class SAEStore:
    def __init__(
            self,
            release_name: str,
            use_error_term: bool,
            device: Literal["cuda", "cpu"],
        ):
        release = get_pretrained_saes_directory()[release_name]
        sae_id_list = list(release.__dict__["saes_map"].values())

        sae_dict = {}
        for sae_id in tqdm(sae_id_list, desc="loading saes"):
            sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
                    release=release_name,
                    sae_id=sae_id,
                    device=device,
                )
            sae.use_error_term = use_error_term
            sae_dict[sae_id] = {"sae": sae, "cfg": cfg_dict}

        self.sae_dict = sae_dict

    def switch_use_error_term(self, use_error_term: bool):
        for sae in self.sae_dict.values():
            sae["sae"].use_error_term = use_error_term

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
    """
    - Decompose head outputs from d_head -> d_model dimensions
    - Retruns head attributions (dot with unembedding matrix) 
    """
    ret_attribs = torch.zeros(12, 12)
    for layer in range(model.cfg.n_layers):    
        decomp_head_out = einsum(
                            cache[f"blocks.{layer}.attn.hook_z"],
                            model.blocks[layer].attn.W_O,
                            "seq n_heads d_head, n_heads d_head d_model -> seq n_heads d_model"
                        )
    
        decomp_head_attribs = einsum(
                            decomp_head_out[-1],
                            model.W_U[:, target_idx],
                            "n_heads d_model, d_model -> n_heads",
                        )
        ret_attribs[layer] = decomp_head_attribs

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
