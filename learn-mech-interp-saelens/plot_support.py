import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import einsum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mc
import matplotlib.colors as mcolors


# ----------------------------------------------------------------------------------------------------------------- #

def plot_feature_vectors(data: torch.Tensor,
                              importance: torch.Tensor = None,
                              figsize_per_plot=(2, 2),
                              max_range=1.5,
                              cmap_name='Blues',
                              annotate=False):
    
    assert data.ndim == 3 and data.shape[2] == 2, "data must be of shape [n_instances, n_vectors, 2]"
    n_instances, n_vectors, _ = data.shape

    # Default importance if not provided
    if importance is None:
        importance = torch.ones((n_instances, n_vectors))
    else:
        assert importance.shape == (n_instances, n_vectors), "importance must match [n_instances, n_vectors]"

    # Normalize importance to [0, 1]
    normed_importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)
    normed_importance = normed_importance * 0.9 + 0.1     # [0, 1] -> [0.1, 1]
    
    cmap = plt.cm.get_cmap(cmap_name)

    # Create subplots
    fig, axs = plt.subplots(1, n_instances, figsize=(figsize_per_plot[0]*n_instances, figsize_per_plot[1]),
                            squeeze=False)
    axs = axs[0]  # Unwrap 1-row grid

    for i in range(n_instances):
        ax = axs[i]
        W = data[i].cpu().numpy()
        imp = normed_importance[i].cpu().numpy()
        colors = cmap(imp)

        # Draw arrows as lines from origin
        lines = np.stack([np.zeros_like(W), W], axis=1)
        ax.add_collection(mc.LineCollection(lines, colors=colors, linewidths=1.5))
        ax.scatter(W[:, 0], W[:, 1], color=colors, s=10)

        if annotate:
            for j, (x, y) in enumerate(W):
                ax.text(x * 1.05, y * 1.05, f"v{j}", fontsize=6)

        # Aesthetics
        ax.set_aspect('equal')
        ax.set_facecolor('#FCFBF8')
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_position('center')

    plt.tight_layout()
    plt.show()
