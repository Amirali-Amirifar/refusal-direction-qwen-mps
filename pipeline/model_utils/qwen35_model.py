import functools
from typing import Sequence

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from pipeline.device import LOCAL_DEVICE
from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.utils import get_orthogonalized_matrix

QWEN35_REFUSAL_TOKENS = [40]  # "I"
QWEN35_RECORDED_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_RECORDED_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def _recorded_revision(model_path: str) -> str | None:
    if model_path.lower() == QWEN35_RECORDED_MODEL.lower():
        return QWEN35_RECORDED_REVISION
    return None


def format_instruction_qwen35(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    output: str | None = None,
    include_trailing_whitespace: bool = True,
) -> str:
    """Render one Qwen3.5 chat turn for generation or loss evaluation."""
    messages = [{"role": "user", "content": instruction}]
    if output is not None:
        messages.append({"role": "assistant", "content": output})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=output is None,
        enable_thinking=False,
    )
    if not include_trailing_whitespace:
        prompt = prompt.rstrip()
    return prompt


def tokenize_instructions_qwen35(
    tokenizer: PreTrainedTokenizerBase,
    instructions: Sequence[str],
    outputs: Sequence[str] | None = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        if len(instructions) != len(outputs):
            raise ValueError("Each instruction must have one corresponding output")
        prompts = [
            format_instruction_qwen35(
                tokenizer,
                instruction=instruction,
                output=output,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen35(
                tokenizer,
                instruction=instruction,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def orthogonalize_qwen35_weights(model, direction: Float[Tensor, "d_model"]):
    with torch.no_grad():
        embedding = model.model.embed_tokens.weight
        embedding.copy_(get_orthogonalized_matrix(embedding, direction))

        for block in model.model.layers:
            if hasattr(block, "self_attn"):
                token_mixer_projection = block.self_attn.o_proj.weight
            elif hasattr(block, "linear_attn"):
                token_mixer_projection = block.linear_attn.out_proj.weight
            else:
                raise AttributeError("Qwen3.5 layer has no supported token mixer")

            token_mixer_projection.copy_(
                get_orthogonalized_matrix(token_mixer_projection.T, direction).T
            )

            projection = block.mlp.down_proj.weight
            projection.copy_(get_orthogonalized_matrix(projection.T, direction).T)


def act_add_qwen35_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(
        bias,
        requires_grad=False,
    )


def get_assistant_eoi_toks(
    tokenizer: PreTrainedTokenizerBase,
    probe_instruction: str = "refusal-direction-boundary-probe",
) -> list[int]:
    """Return the template-derived user-to-assistant boundary tokens."""
    prompt = format_instruction_qwen35(tokenizer, probe_instruction)
    instruction_end = prompt.find(probe_instruction)
    if instruction_end == -1:
        raise ValueError("Probe instruction was not found in the rendered chat template")

    boundary = prompt[instruction_end + len(probe_instruction) :]
    boundary_tokens = tokenizer.encode(boundary, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    if not boundary_tokens or prompt_tokens[-len(boundary_tokens) :] != boundary_tokens:
        raise ValueError("Could not derive a stable Qwen3.5 assistant boundary")

    return boundary_tokens


class Qwen35Model(ModelBase):
    def _load_model(self, model_path, dtype=torch.bfloat16):
        revision = _recorded_revision(model_path)
        revision_kwargs = {"revision": revision} if revision is not None else {}
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=False,
            **revision_kwargs,
        )
        model = model.to(LOCAL_DEVICE).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        revision = _recorded_revision(model_path)
        revision_kwargs = {"revision": revision} if revision is not None else {}
        tokenizer = AutoTokenizer.from_pretrained(model_path, **revision_kwargs)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen35,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return get_assistant_eoi_toks(self.tokenizer)

    def _get_refusal_toks(self):
        # During adaptation, token 40 was the greedy first token for 35/40
        # sampled harmful prompts and was uncommon on the harmless sample.
        return QWEN35_REFUSAL_TOKENS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [
                block.self_attn if hasattr(block, "self_attn") else block.linear_attn
                for block in self.model.model.layers
            ]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model.model.layers])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen35_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_qwen35_weights,
            direction=direction,
            coeff=coeff,
            layer=layer,
        )
