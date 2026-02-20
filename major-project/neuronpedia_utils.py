from IPython.display import HTML, IFrame, display

import circuitsvis as cv
import plotly.express as px

import sae_lens
from sae_lens import SAE, ActivationsStore, HookedSAETransformer, LanguageModelSAERunnerConfig
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

# ------------------------------------------------------------------------------------------------------------------------------ #

def display_dashboard(
        neuronpedia_id:str,
        latent_idx:int,
        width:int=950,
        height:int=450,
):
    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    print(url)
    display(IFrame(url, width=width, height=height))


