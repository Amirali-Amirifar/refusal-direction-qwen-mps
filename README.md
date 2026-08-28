# Refusal directions on Qwen3.5-0.8B

This repository contains a reproduction of [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717) on [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). 

A **refusal direction** is a vector $\hat{\mathbf{r}} \in \mathbb{R}^{d_{\text{model}}}$ such that we can remove the model's ability to refuse by erasing this direction from its activations:

$$ \mathbf{x}' \leftarrow \mathbf{x} - \hat{\mathbf{r}}\hat{\mathbf{r}}^\top\mathbf{x} $$

This operation is performed at every activation across all layers. This project is based on the [authors' original code](https://github.com/andyrdt/refusal_direction), adapted to run the experiment end-to-end on an Apple Silicon Mac and without dependencies to API providers.


## Results

The run selected layer 11 at prompt position -8. Seven of the 216 candidates passed
the selection thresholds used by the original implementation; all seven were in
layers 8–11.

| Measurement | Baseline | Ablation | Activation addition |
| --- | ---: | ---: | ---: |
| Harmful prompts: substring ASR | 0.27 | 0.99 | 0.99 (negative) |
| Harmless prompts: non-refusal rate | 0.98 | — | 0.03 (positive) |
| CE loss on harmless completions | 0.580 | 0.605 | 0.956 (negative) |

The saved direction, plots, completions, and per-example evaluations are in
[`pipeline/runs/qwen3.5-0.8b`](pipeline/runs/qwen3.5-0.8b/).

Here, substring ASR is simply the fraction of responses that do not contain one of
the refusal phrases from the original evaluation code. It is useful for a local
reproduction, but it should not be confused with a semantic judgment of whether a
response is harmful or useful.

## Running the pipeline

Install [`uv`](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/Amirali-Amirifar/refusal-direction-qwen-mps.git
cd refusal-direction-qwen-mps
uv sync --frozen
uv run --frozen python -m pipeline.run_pipeline --model_path Qwen/Qwen3.5-0.8B
```

To chat with the model normally:

```bash
uv run --frozen python chat_qwen35.py
```

To apply the saved refusal-direction ablation:

```bash
uv run --frozen python chat_qwen35.py --ablated
```

## Credit

The method and original implementation are by Andy Arditi, Oscar Obeso, Aaquib
Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda:

- [Paper](https://arxiv.org/abs/2406.11717)
- [Original repository](https://github.com/andyrdt/refusal_direction)
- [Original explanatory post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## License

This repository keeps the original [Apache License 2.0](LICENSE).
