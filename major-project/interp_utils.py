import torch
from torch import Tensor
import torch.nn.functional as F
from einops import einsum
import einops
from einops import einsum
import seaborn as sns
from functools import partial

import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache

import sae_lens
from sae_lens import SAE, ActivationsStore, HookedSAETransformer, LanguageModelSAERunnerConfig
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from tqdm.auto import tqdm
from jaxtyping import Int, Float
from typing import List, Tuple, Optional, Literal, Any, Callable
import numpy as np
from transformer_lens import utils

from rich import print as rprint
from rich.table import Table

# ------------------------------------------------------------------------------------------------------------------------------ #

# -- Helper Functions[fetch_max_activating_examples] -- #
def get_k_largest_indices(
    x: Float[Tensor, "batch seq"],
    k: int,
    buffer: int = 0,
    no_overlap: bool = True,
) -> Int[Tensor, "k 2"]:

    assert buffer * 2 < x.size(1), "Buffer is too large for the sequence length"
    assert not no_overlap or k <= x.size(0), (
        "Not enough sequences to have a different token in each sequence"
    )

    if buffer > 0:
        x = x[:, buffer:-buffer]

    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer

    if no_overlap:
        unique_indices = torch.empty((0, 2), device=x.device).long()
        while len(unique_indices) < k:
            unique_indices = torch.cat(
                (unique_indices, torch.tensor([[rows[0], cols[0]]], device=x.device))
            )
            is_overlapping_mask = (rows == rows[0]) & ((cols - cols[0]).abs() <= buffer)
            rows = rows[~is_overlapping_mask]
            cols = cols[~is_overlapping_mask]
        return unique_indices

    return torch.stack((rows, cols), dim=1)[:k]


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, "k *buffer_x2_plus1"]:

    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        rows = einops.repeat(rows, "k -> k buffer", buffer=buffer * 2 + 1)
        cols[cols < buffer] = buffer
        cols[cols > x.size(1) - buffer - 1] = x.size(1) - buffer - 1
        cols = einops.repeat(cols, "k -> k buffer", buffer=buffer * 2 + 1) + torch.arange(
            -buffer, buffer + 1, device=cols.device
        )
    return x[rows, cols]

def display_top_seqs(data: list[tuple[float, list[str], int]]):
    table = Table("Act", "Sequence", title="Max Activating Examples", show_lines=True)
    for act, str_toks, seq_pos in data:
        formatted_seq = (
            "".join(
                [
                    f"[b u green]{str_tok}[/]" if i == seq_pos else str_tok
                    for i, str_tok in enumerate(str_toks)
                ]
            )
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)


# -- Fetch Max Activating Examples -- #
def fetch_max_activating_examples(
    model: HookedSAETransformer,
    sae: SAE,
    latent_idx: int,
    act_store: ActivationsStore = None,
    total_batches: int = 100,
    k: int = 10,
    buffer: int = 10,
) -> list[tuple[float, list[str], int]]:
    
    if act_store == None:
        print("bulding activation store...")
        act_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            dataset="monology/pile-uncopyrighted",  
            streaming=True,
            store_batch_size_prompts=16,
            n_batches_in_buffer=32,
            device="cuda",
        )

    sae_acts_post_hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
    data = []

    for _ in tqdm(range(total_batches), desc="Computing activations"):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=int(sae.cfg.metadata.hook_name.split(".")[1]) + 1,
            names_filter=[sae_acts_post_hook_name],
        )
        acts = cache[sae_acts_post_hook_name][..., latent_idx]

        # Get largest indices, get the corresponding max acts, and get the surrounding indices
        k_largest_indices = get_k_largest_indices(acts, k=k, buffer=buffer)
        tokens_with_buffer = index_with_buffer(tokens, k_largest_indices, buffer=buffer)
        str_toks = [model.to_str_tokens(toks) for toks in tokens_with_buffer]
        top_acts = index_with_buffer(acts, k_largest_indices).tolist()
        data.extend(list(zip(top_acts, str_toks, [buffer] * len(str_toks))))

    return sorted(data, key=lambda x: x[0], reverse=True)[:k]


def display_max_activating_examples(
    model: HookedSAETransformer,
    sae: SAE,
    latent_idx: int,
    act_store: ActivationsStore = None,
    total_batches: int = 100,
    k: int = 10,
    buffer: int = 10,
    ret_data: bool = False,
):
    data = fetch_max_activating_examples(
        model=model,
        sae=sae,
        latent_idx=latent_idx,
        act_store=act_store,
        total_batches=total_batches,
        k=k,
        buffer=buffer,
    )
    display_top_seqs(data)
    if ret_data:
        return data


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
            sae_dict[sae_id] = {
                "sae": sae,
                "cfg": cfg_dict,
            }

        self.sae_cfg = cfg_dict
        self.sae_dict = sae_dict
        self.device = device

    def switch_use_error_term(self, use_error_term: bool):
        for sae in self.sae_dict.values():
            sae["sae"].use_error_term = use_error_term
    
    def get_sae_list(self, item:Literal["cfg", "sae"]):
        return [sae[item] for sae in list(self.sae_dict.values())]


    # -- Mean Activation -- #
    @staticmethod
    def _get_token_pos(
        model: HookedSAETransformer,
        sample: str,
        target_token: str,
    ):
        for i, token in enumerate(model.to_str_tokens(sample)):
            if token == target_token:
                return i
        return None

    def mean_sae_decoder_activation(
        self,
        sae_model: HookedSAETransformer,
        samples: list[str],
        target_id: int,
        layers: list[int] | None = None,
        pre_target: bool = True,
    ):
        """
        Compute the mean SAE decoder activation at the (pre-)target token
        position across samples and layers.
        """
        target_token = sae_model.to_single_str_token(target_id)

        saes = self.get_sae_list(item="sae")
        hook_names = [f"{hook_name}.hook_sae_acts_post" for hook_name in self.sae_dict.keys()][:-1]

        _, cache = sae_model.run_with_cache_with_saes(
            samples,
            saes=saes,
            names_filter=hook_names,
        )
        n_layers = sae_model.cfg.n_layers
        d_sae = self.sae_cfg["d_sae"]

        target_cache = torch.zeros((len(samples), n_layers, d_sae), device=self.device)
        for sample_idx, sample in enumerate(samples):
            target_pos = self._get_token_pos(sae_model, sample, target_token) - int(pre_target)
            assert target_pos is not None, (f"Target token {target_token!r} not found "f"in sample {sample_idx}")

            for layer in range(n_layers):
                layer_name = (f"blocks.{layer}.hook_resid_pre.hook_sae_acts_post")
                target_cache[sample_idx, layer] = cache[layer_name][sample_idx, target_pos, :]

        # validate layers        
        if layers is not None:
            n_layers = target_cache.shape[1]
        
            bad_layers = [l for l in layers if l < 0 or l >= n_layers]
            if bad_layers:
                raise ValueError(
                    f"Invalid layer indices {bad_layers}. "
                    f"Valid range is [0, {n_layers - 1}]"
                )

        # layer selection + reduction
        if layers is not None:
            if len(layers) == 1:
                return torch.squeeze(torch.mean(target_cache[:, layers, :], dim=0))     
            return torch.mean(target_cache[:, layers, :], dim=0)
    
        return torch.mean(target_cache, dim=0)
    
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

# ---------------------------------------------------------------------------------------------------------------- #

def show_topk_preds(model, prompt, k=10):
    
    logits = model(prompt, return_type="logits")
    top_logits, top_ids = torch.topk(logits[0, -1, :], k=k)
    top_probs = torch.softmax(top_logits, dim=-1)

    for prob, idx in zip(top_probs, top_ids):
        prob_val = prob.item()
        token = model.to_single_str_token(idx.item())
        print(f"PROB: {prob_val * 100:.2f}%  TOKEN: |{token}|")
        

def show_token_scores(model, prompt, target_id):
    
    logits = model(prompt, return_type="logits")
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    
    target_prob = probs[target_id].item()
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1

    print(f"TOKEN: |{model.to_single_str_token(target_id)}| RANK: {rank}, PROB: {target_prob}")

# ---------------------------------------------------------------------------------------------------------------- #

