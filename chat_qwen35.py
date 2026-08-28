"""Interactive Qwen3.5-0.8B chat with optional directional ablation."""

import argparse
from pathlib import Path

import torch

from pipeline.model_utils.qwen35_model import Qwen35Model
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_DIRECTION = Path("pipeline/runs/qwen3.5-0.8b/direction.pt")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with Qwen3.5-0.8B locally on Apple Silicon."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name or local path")
    parser.add_argument(
        "--ablated",
        action="store_true",
        help="remove the saved refusal direction during generation",
    )
    parser.add_argument(
        "--direction",
        type=Path,
        default=DEFAULT_DIRECTION,
        help="path to the saved refusal direction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    model_base = Qwen35Model(args.model)
    model = model_base.model
    tokenizer = model_base.tokenizer

    forward_pre_hooks, forward_hooks = [], []
    if args.ablated:
        direction = torch.load(args.direction, map_location="cpu", weights_only=True)
        forward_pre_hooks, forward_hooks = get_all_direction_ablation_hooks(
            model_base,
            direction,
        )
        print(f"[ablated] using refusal direction from {args.direction}")
        print("Generated responses may contain unsafe or offensive content.")

    history = []
    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user.strip():
            continue

        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode(), add_hooks(forward_pre_hooks, forward_hooks):
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
            )

        generated_tokens = output[0][inputs["input_ids"].shape[-1] :]
        reply = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()
        print("\nQwen:", reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
