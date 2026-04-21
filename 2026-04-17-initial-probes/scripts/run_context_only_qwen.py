"""Run a context-only Qwen2.5 baseline over the synthetic trajectories.

This script does not implement persistent memory. It serializes each trajectory
into plain text, places that text in the model context, asks one evaluation
query at a time, and prints the model answer beside the expected answer.

Install dependencies first:

    pip install torch transformers accelerate

Then run from the repository root:

    python 2026-04-17-initial-probes/scripts/run_context_only_qwen.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATA_PATH = EXPERIMENT_ROOT / "data/examples/synthetic_trajectories.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def render_trajectory_context(trajectory: dict[str, Any]) -> str:
    """Convert one structured trajectory into plain text context for the LM."""
    entity_names = {entity["id"]: entity["name"] for entity in trajectory["entities"]}

    def label(value: str) -> str:
        return entity_names.get(value, value)

    lines: list[str] = []
    lines.append(f"Trajectory: {trajectory['title']}")
    lines.append(trajectory["description"])
    lines.append("")

    lines.append("Entities:")
    for entity in trajectory["entities"]:
        aliases = ", ".join(entity.get("aliases", [])) or "none"
        lines.append(
            f"- {entity['id']}: {entity['name']} "
            f"({entity['type']}), aliases: {aliases}"
        )

    lines.append("")
    lines.append("Initial facts:")
    for fact in trajectory["initial_facts"]:
        timestamp = fact["metadata"]["timestamp"]
        scope = fact["metadata"]["scope"]
        lines.append(
            f"- At {timestamp} [{scope}], "
            f"{label(fact['subject'])} {fact['predicate']} {label(fact['object'])}."
        )

    lines.append("")
    lines.append("Updates:")
    for update in trajectory["updates"]:
        timestamp = update["metadata"]["timestamp"]
        scope = update["metadata"]["scope"]
        operation = update["operation"]
        lines.append(
            f"- At {timestamp} [{scope}], {operation}: "
            f"{label(update['subject'])} {update['predicate']} {label(update['object'])}."
        )

    return "\n".join(lines)


def ask_model(
    model: Any,
    tokenizer: Any,
    context: str,
    query: dict[str, Any],
    max_new_tokens: int,
) -> str:
    """Ask one benchmark query using the trajectory as plain context."""
    question = query["question"]
    belief_type = query["belief_type"]
    query_time = query["query_time"]
    belief_time = query["belief_time"]

    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions. "
                # "Use only the provided trajectory context. "
                # "Pay attention to timestamps, scopes, and revisions. "
                # "For current-belief questions, answer with the latest valid belief. "
                # "For historical-belief questions, answer about the requested earlier time. "
                # "Answer briefly with only the answer unless clarification is necessary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question metadata:\n"
                f"- belief_type: {belief_type}\n"
                f"- query_time: {query_time}\n"
                f"- belief_time: {belief_time}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    import torch

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    answer_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()


def run_context_only_baseline(
    model_name: str,
    data_path: Path,
    max_new_tokens: int,
    limit: int | None,
    show_context: bool,
) -> None:
    """Run Qwen over benchmark trajectories with full trajectory context."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = load_json(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    trajectories = data["trajectories"]
    if limit is not None:
        trajectories = trajectories[:limit]

    for trajectory in trajectories:
        context = render_trajectory_context(trajectory)

        print(f"\n=== {trajectory['id']} ===")
        print(trajectory["title"])

        if show_context:
            print("\n--- Context ---")
            print(context)
            print("--- End Context ---")

        for query in trajectory["evaluation_queries"]:
            answer = ask_model(
                model=model,
                tokenizer=tokenizer,
                context=context,
                query=query,
                max_new_tokens=max_new_tokens,
            )

            print(f"\nQ: {query['question']}")
            print(f"belief_type: {query['belief_type']}")
            print(f"model: {answer}")
            print(f"expected: {query['expected_answer']}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"HF model name or local path. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Benchmark JSON path. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per answer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of trajectories to run for quick smoke tests.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the rendered trajectory context before asking its queries.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_context_only_baseline(
        model_name=args.model,
        data_path=args.data,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        show_context=args.show_context,
    )


if __name__ == "__main__":
    main()
