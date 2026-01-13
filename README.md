## llm-interp-practice

This repository is dedicated to **practicing mechanistic interpretability (mech-interp) of LLMs**.  

The work is mainly inspired by:  
- [Arena Chapter 1: Transformer Interpretability](https://arena-chapter1-transformer-interp.streamlit.app/)  
- [Arena Chapter 1: Transformers](https://arena-ch1-transformers.streamlit.app/)  

The goal is to **solve exercises and reproduce experiments** from these resources, primarily using the **SAELens** and **TransformerLens** libraries.

---

### Structure

#### `learn-mech-interp-saelens`

Focused on **Sparse Autoencoder (SAE) based interpretability** experiments.

Notebooks:
- `superposition-in-non-previleged-basis.ipynb` — exploring superposition of features in non-privileged bases  
- `intro-to-sae-interpretability.ipynb` — basic SAE experiments and interpretability techniques  

Supporting files:
- `plot_support.py` — helper functions for visualizations  
- `test_support.py` — testing functions for experiment validation  

---

#### `learn-mech-interp-transformerlens`

Focused on **TransformerLens-based experiments**, working directly with transformer internals.

Notebooks:
- `ch1-1-inputs&outputs.ipynb` — working with embeddings, unembedding vectors, and outputs  
- `ch1-2-implement-transformer.ipynb` — building a transformer from scratch  
- `ch1-4-sampling_methods.ipynb` — exploring different sampling methods: greedy, temperature-based, top-k, and beam search  
- `ch2-1-transformerlens-intro.ipynb` — core ideas and usage of important TransformerLens classes  
- `ch2-2-finding-induction-heads.ipynb` — detecting induction heads by observing attention patterns  
- `ch2-3-transformerlens-hooks.ipynb` — using hooks to read and intervene on activations  
- `ch2-4-reverse_engineering-induction-circuit.ipynb` — reproducing experiments from the induction circuit paper (currently incomplete, will revisit later)

---

### Goals & Status

- **Primary goal:** Understand LLM internals via hands-on mechanistic interpretability exercises  
- **Status:**  
  - SAELens notebooks: working and mostly complete  
  - TransformerLens notebooks: mostly complete, some advanced concepts (like the induction-circuit) remain unfinished  
  - Experiments focus on **observing, visualizing, and intervening on model activations**  

