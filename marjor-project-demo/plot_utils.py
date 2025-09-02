import torch
from torch import Tensor
import torch.nn.functional as F
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
from einops import einsum

from jaxtyping import Int, Float
from typing import List, Tuple, Optional, Literal
import numpy as np
from transformer_lens import utils

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import numpy as np

from interp_utils import get_layer_attributions, decomposed_head_attribs

# ---------------------------------------------------------------------------------------------------------------- #

def visualize_transformer_attributions(
    head_attributions,
    mlp_attributions,
    input_token,
    output_token,
    show_node_labels=True,
    use_normalized_colors=True,
    show_colorbar=True,
    save_path=None  # New parameter to optionally save the figure
):
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib import cm

    num_layers = head_attributions.shape[0]
    num_heads = head_attributions.shape[1]

    # Normalize if requested
    if use_normalized_colors:
        head_attr_vals = (head_attributions - head_attributions.min()) / (head_attributions.max() - head_attributions.min() + 1e-8)
        mlp_attr_vals = (mlp_attributions - mlp_attributions.min()) / (mlp_attributions.max() - mlp_attributions.min() + 1e-8)
        head_norm_display = True
        mlp_norm_display = True
    else:
        head_attr_vals = head_attributions
        mlp_attr_vals = mlp_attributions
        head_norm_display = False
        mlp_norm_display = False

    # Create graph
    G = nx.DiGraph()
    pos = {}
    node_colors = []
    node_labels = {}

    y_spacing = 1.5
    x_spacing = 4.0

    G.add_node("INPUT", layer=0)
    pos["INPUT"] = (0, 0)
    node_colors.append("lightgreen")
    node_labels["INPUT"] = input_token

    current_x = x_spacing
    head_node_colors = []
    mlp_node_colors = []

    for layer in range(num_layers):
        for head in range(num_heads):
            node_name = f"L{layer+1}H{head}"
            G.add_node(node_name, layer=layer+1)
            y = (head - num_heads/2) * y_spacing
            pos[node_name] = (current_x, y)

            attr_value = head_attributions[layer, head]
            color_value = head_attr_vals[layer, head]
            cmap = cm.Blues
            color = cmap(color_value)
            head_node_colors.append(color)
            node_colors.append(color)

            if show_node_labels:
                node_labels[node_name] = f"{node_name}\n{attr_value:.3f}"
            else:
                node_labels[node_name] = f"{attr_value:.3f}"

            if layer == 0:
                G.add_edge("INPUT", node_name)
            else:
                prev_mlp = f"L{layer}MLP"
                G.add_edge(prev_mlp, node_name)

        mlp_node = f"L{layer+1}MLP"
        G.add_node(mlp_node, layer=layer+1.5)
        pos[mlp_node] = (current_x + x_spacing/2, 0)

        attr_value = mlp_attributions[layer]
        color_value = mlp_attr_vals[layer]
        cmap = cm.Oranges
        color = cmap(color_value)
        mlp_node_colors.append(color)
        node_colors.append(color)

        if show_node_labels:
            node_labels[mlp_node] = f"{mlp_node}\n{attr_value:.3f}"
        else:
            node_labels[mlp_node] = f"{attr_value:.3f}"

        for head in range(num_heads):
            head_node = f"L{layer+1}H{head}"
            G.add_edge(head_node, mlp_node)

        current_x += x_spacing

    G.add_node("OUTPUT", layer=num_layers + 1)
    pos["OUTPUT"] = (current_x, 0)
    node_colors.append("lightcoral")
    node_labels["OUTPUT"] = output_token

    last_mlp = f"L{num_layers}MLP"
    G.add_edge(last_mlp, "OUTPUT")

    # Create figure
    fig = plt.figure(figsize=(35, 15))
    grid_spec = fig.add_gridspec(1, 5, width_ratios=[20, 0.25, 0.25, 0.25, 0.25])
    ax_main = fig.add_subplot(grid_spec[0, 0])

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, edgecolors='black', ax=ax_main)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', edge_color='gray', ax=ax_main)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='black', ax=ax_main)

    ax_main.set_title(f"Attribution Flow: '{input_token}' → '{output_token}'", fontsize=14)
    ax_main.axis("off")

    if show_colorbar:
        norm = plt.Normalize(vmin=head_attributions.min(), vmax=head_attributions.max()) if not head_norm_display else plt.Normalize(0, 1)
        sm_head = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
        sm_head.set_array([])
        cbar_ax1 = fig.add_subplot(grid_spec[0, 1])
        fig.colorbar(sm_head, cax=cbar_ax1)
        cbar_ax1.set_title("Heads", fontsize=10)

        norm = plt.Normalize(vmin=mlp_attributions.min(), vmax=mlp_attributions.max()) if not mlp_norm_display else plt.Normalize(0, 1)
        sm_mlp = plt.cm.ScalarMappable(cmap=cm.Oranges, norm=norm)
        sm_mlp.set_array([])
        cbar_ax2 = fig.add_subplot(grid_spec[0, 2])
        fig.colorbar(sm_mlp, cax=cbar_ax2)
        cbar_ax2.set_title("MLP", fontsize=10)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------------------------------------------- #

def get_and_visualize_model_attributions(
    model: HookedTransformer,
    input_str: str,
    target_idx: int,
    show_node_labels: bool=True,
    use_normalized_colors: bool=True,
    show_colorbar: bool=True,
    save_path: bool="test.png",
):
    
    logits, cache = model.run_with_cache(input_str, remove_batch_dim=True)
    
    dcm_head_attribs = decomposed_head_attribs(model, cache, target_idx)
    mlp_layer_attribs = get_layer_attributions(model, "mlp", cache, target_idx)
    
    detached_dcm_head_attribs = dcm_head_attribs.detach().cpu().numpy()
    detached_mlp_layer_attribs = mlp_layer_attribs.detach().cpu().numpy()

    visualize_transformer_attributions(
        head_attributions=detached_dcm_head_attribs,
        mlp_attributions=detached_mlp_layer_attribs,
        input_token=model.to_str_tokens(input_str)[-1], 
        output_token=model.to_string(target_idx),
        show_node_labels=show_node_labels,
        use_normalized_colors=use_normalized_colors,
        show_colorbar=show_colorbar,
        save_path=save_path,
    )

# ---------------------------------------------------------------------------------------------------------------- #
