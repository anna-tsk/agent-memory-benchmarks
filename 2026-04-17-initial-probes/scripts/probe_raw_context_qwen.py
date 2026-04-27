"""Probe Qwen2.5 with raw observations and no expected answers.

This is not a benchmark runner. It is an exploratory script for seeing how an
instruct model resolves sequential claims when the context does not include
explicit update labels such as `revise`, `supersedes`, or gold behavior.

Install dependencies first:

    pip install torch transformers accelerate

Then run from the repository root:

    python 2026-04-17-initial-probes/scripts/probe_raw_context_qwen.py --limit 1 --show-context

Use dialogue mode when later questions should see earlier model answers:

    python 2026-04-17-initial-probes/scripts/probe_raw_context_qwen.py --limit 1 --dialogue-mode
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATA_PATH = EXPERIMENT_ROOT / "data/examples/raw_observation_probes_v2.json"
DEFAULT_RESPONSE_FORMAT = [
    "Direct answer",
    "Possible interpretations",
    "Assumption(s) made",
    "Missing information or useful follow-up",
    "Memory handling",
]
CONVERSATIONAL_RESPONSE_FORMAT = [
    "Direct answer",
    "Possible interpretations",
    "Assumption(s) made",
    "Clarifying question",
    "Memory handling",
]
DEFAULT_RUNS_DIR = EXPERIMENT_ROOT / "runs"


class TeeLogger:
    """Print lines to stdout and optionally mirror them to a log file."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8") if path is not None else None

    def close(self) -> None:
        """Close the log file if one is open."""
        if self._handle is not None:
            self._handle.close()

    def print(self, *values: object, sep: str = " ", end: str = "\n") -> None:
        """Print to stdout and the optional log file."""
        text = sep.join(str(value) for value in values) + end
        sys.stdout.write(text)
        sys.stdout.flush()
        if self._handle is not None:
            self._handle.write(text)
            self._handle.flush()


def slugify(value: str) -> str:
    """Convert a user-provided log name into a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return slug or "probe_run"


def make_log_path(log_name: str | None, runs_dir: Path) -> Path | None:
    """Build a timestamped run log path, if logging was requested."""
    if log_name is None:
        return None
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return runs_dir / f"{slugify(log_name)}_{timestamp}.txt"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def render_probe_context(probe: dict[str, Any]) -> str:
    """Render raw timestamped observations without interpretation labels."""
    lines: list[str] = []
    lines.append(f"Scenario: {probe['title']}")
    lines.append("")
    lines.append("All observations are statements from the same speaker.")
    lines.append("")
    lines.append("Observations:")
    for observation in probe["observations"]:
        lines.append(f"- {observation['timestamp']} | {observation['content']}")
    return "\n".join(lines)


def normalize_question(question: str | dict[str, Any]) -> tuple[str, str]:
    """Return a display label and prompt text for a question entry."""
    if isinstance(question, str):
        return "question", question
    return question.get("id", "question"), question["text"]


def get_response_format(condition: str, data: dict[str, Any]) -> list[str] | None:
    """Return the output sections for the selected prompt condition."""
    if condition == "minimal":
        return None
    if condition == "analytic":
        return data.get("response_format", DEFAULT_RESPONSE_FORMAT)
    if condition == "conversational":
        return CONVERSATIONAL_RESPONSE_FORMAT
    raise ValueError(f"Unknown condition: {condition}")


def get_conditions(condition: str) -> list[str]:
    """Expand the requested prompt condition into concrete conditions to run."""
    if condition == "all":
        return ["minimal", "analytic", "conversational"]
    return [condition]


def format_question(question: str, response_format: list[str] | None) -> str:
    """Wrap a probe question in a consistent qualitative response format."""
    if response_format is None:
        return question

    numbered_sections = "\n".join(
        f"{index}. {section}:" for index, section in enumerate(response_format, start=1)
    )
    return (
        f"{question}\n\n"
        "Use exactly this format:\n"
        f"{numbered_sections}"
    )


def ask_model(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    """Generate one answer from a chat message history."""
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


def make_single_turn_messages(context: str, question: str) -> list[dict[str, str]]:
    """Build a fresh single-question prompt."""
    return [
        {
            "role": "system",
            "content": "You answer questions.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]


def make_dialogue_messages(context: str) -> list[dict[str, str]]:
    """Build the initial message history for dialogue-style probing."""
    return [
        {
            "role": "system",
            "content": "You answer questions.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nI will ask questions about this context.",
        },
        {
            "role": "assistant",
            "content": "Understood.",
        },
    ]


def filter_probes(
    probes: list[dict[str, Any]],
    probe_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter probes by explicit IDs, preserving the JSON file order."""
    if not probe_ids:
        return probes

    requested = set(probe_ids)
    found = {probe["id"] for probe in probes if probe["id"] in requested}
    missing = sorted(requested - found)
    if missing:
        available = ", ".join(probe["id"] for probe in probes)
        raise ValueError(
            f"Unknown probe id(s): {', '.join(missing)}. "
            f"Available probe ids: {available}"
        )

    return [probe for probe in probes if probe["id"] in requested]


def run_probe(
    model_name: str,
    data_path: Path,
    max_new_tokens: int,
    limit: int | None,
    probe_ids: list[str] | None,
    show_context: bool,
    show_prompt: bool,
    dialogue_mode: bool,
    condition: str,
    log_name: str | None,
    runs_dir: Path,
) -> None:
    """Run exploratory raw-observation probes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log_path = make_log_path(log_name, runs_dir)
    logger = TeeLogger(log_path)
    log = logger.print

    try:
        if log_path is not None:
            log(f"Logging to: {log_path}")
            log(f"model: {model_name}")
            log(f"data: {data_path}")
            log(f"max_new_tokens: {max_new_tokens}")
            log(f"limit: {limit}")
            log(f"probe_ids: {probe_ids}")
            log(f"show_context: {show_context}")
            log(f"show_prompt: {show_prompt}")
            log(f"dialogue_mode: {dialogue_mode}")
            log(f"condition: {condition}")
            log("")

        data = load_json(data_path)
        conditions = get_conditions(condition)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        probes = filter_probes(data["probes"], probe_ids)
        if limit is not None:
            probes = probes[:limit]

        for probe in probes:
            context = render_probe_context(probe)

            log(f"\n=== {probe['id']} ===")
            log(probe["title"])

            if show_context:
                log("\n--- Context ---")
                log(context)
                log("--- End Context ---")

            for active_condition in conditions:
                response_format = get_response_format(active_condition, data)
                log(f"\n--- Condition: {active_condition} ---")

                messages = make_dialogue_messages(context) if dialogue_mode else None

                for question_entry in probe["questions"]:
                    question_id, question_text = normalize_question(question_entry)
                    formatted_question = format_question(question_text, response_format)
                    if dialogue_mode:
                        assert messages is not None
                        messages.append({"role": "user", "content": formatted_question})
                        if show_prompt:
                            log("\n--- Prompt Messages ---")
                            for index, message in enumerate(messages, start=1):
                                log(
                                    f"[{index}] {message['role']}:\n"
                                    f"{message['content']}"
                                )
                            log("--- End Prompt Messages ---")
                        answer = ask_model(
                            model=model,
                            tokenizer=tokenizer,
                            messages=messages,
                            max_new_tokens=max_new_tokens,
                        )
                        messages.append({"role": "assistant", "content": answer})
                    else:
                        prompt_messages = make_single_turn_messages(
                            context, formatted_question
                        )
                        if show_prompt:
                            log("\n--- Prompt Messages ---")
                            for index, message in enumerate(prompt_messages, start=1):
                                log(
                                    f"[{index}] {message['role']}:\n"
                                    f"{message['content']}"
                                )
                            log("--- End Prompt Messages ---")
                        answer = ask_model(
                            model=model,
                            tokenizer=tokenizer,
                            messages=prompt_messages,
                            max_new_tokens=max_new_tokens,
                        )

                    log(f"\nQ [{question_id}]: {question_text}")
                    log(f"condition: {active_condition}")
                    log(f"model: {answer}")
    finally:
        logger.close()


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
        help=f"Probe JSON path. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per answer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of probes to run.",
    )
    parser.add_argument(
        "--probe-id",
        action="append",
        default=None,
        help=(
            "Run only a specific probe id. Can be passed multiple times, "
            "for example --probe-id probe_sister --probe-id probe_doctor."
        ),
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the rendered raw context before asking probe questions.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the exact chat messages sent to the tokenizer for each question.",
    )
    parser.add_argument(
        "--dialogue-mode",
        action="store_true",
        help="Append each model answer to the chat before asking the next question.",
    )
    parser.add_argument(
        "--condition",
        choices=["minimal", "analytic", "conversational", "all"],
        default="analytic",
        help=(
            "Prompt framing condition: minimal asks only the question; analytic "
            "uses Missing information or useful follow-up; conversational uses "
            "Clarifying question; all runs the three conditions sequentially. "
            "Default: analytic."
        ),
    )
    parser.add_argument(
        "--log-name",
        default=None,
        help="Optional name for saving this run to runs/<name>_<timestamp>.txt.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory for --log-name outputs. Default: {DEFAULT_RUNS_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_probe(
        model_name=args.model,
        data_path=args.data,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        probe_ids=args.probe_id,
        show_context=args.show_context,
        show_prompt=args.show_prompt,
        dialogue_mode=args.dialogue_mode,
        condition=args.condition,
        log_name=args.log_name,
        runs_dir=args.runs_dir,
    )


if __name__ == "__main__":
    main()
