import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tabulate import tabulate
import numpy
import random
from pprint import pprint
from einops import einsum

from typing import Any, Callable, Literal, TypeAlias
from jaxtyping import Float, Int

from IPython.display import HTML, IFrame, display
from datasets import load_dataset
from huggingface_hub import hf_hub_download

import circuitsvis as cv
import sae_lens
from sae_lens import SAE, ActivationsStore, HookedSAETransformer, LanguageModelSAERunnerConfig
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy

# ------------------------------------------------------------------------------------------------------------------ #

def show_topk_preds(model, prompt, k=10):
    
    logits = model(prompt, return_type="logits")
    top_logits, top_ids = torch.topk(logits[0, -1, :], k=k)
    top_probs = torch.softmax(top_logits, dim=-1)

    for prob, idx in zip(top_probs, top_ids):
        prob_val = prob.item()
        token = model.to_single_str_token(idx.item())
        print(f"PROB: {prob_val * 100:.2f}%  TOKEN: |{token}|")
        
# ------------------------------------------------------------------------------------------------------------------ #

def show_token_scores(model, prompt, target_id):
    
    logits = model(prompt, return_type="logits")
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    
    target_prob = probs[target_id].item()
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1

    print(f"TOKEN: |{model.to_single_str_token(target_id)}| RANK: {rank}, PROB: {target_prob}")

# ------------------------------------------------------------------------------------------------------------------ #
