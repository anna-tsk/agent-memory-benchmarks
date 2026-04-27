"""
Embedding noise-floor diagnostic for Qwen2.5-3B-Instruct.

Pooling methods:
  last_token    — hidden state of the final (non-padding) token, last layer
  mean          — mean-pooled over non-padding tokens, last layer
  mean_midlayer — mean-pooled over non-padding tokens, layer 18 of 36
"""

import os
import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── configuration ────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-3B-Instruct"
MID_LAYER    = 18          # 0-indexed; Qwen2.5-3B has 36 transformer layers
OUT_DIR      = "diagnostics"
METHODS      = ["last_token", "mean", "mean_midlayer"]

probes = {
    "marie_best_friend":           "Marie is my best friend.",
    "tina_best_friend":            "Tina is my best friend.",
    "marie_enemy":                 "Marie is my enemy.",
    "marie_best_friend_paraphrase":"My best friend is Marie.",
    "bob_best_friend":             "Bob is my best friend.",
    "unrelated_weather":           "It rained heavily yesterday.",
    "unrelated_tech":              "The new processor has eight cores.",
    "marie_sister":                "Marie is my sister.",
    "tina_sister":                 "Tina is my sister.",
    "anna_la":                     "Anna lives in Los Angeles.",
    "anna_sf":                     "Anna lives in San Francisco.",
    "anna_la_past":                "Anna used to live in Los Angeles.",
}

# pairs we care about most, with human-readable labels
FOCAL_PAIRS = [
    ("marie_best_friend", "tina_best_friend",
     "same predicate, different name"),
    ("marie_best_friend", "marie_enemy",
     "same name, opposite predicate"),
    ("marie_best_friend", "marie_best_friend_paraphrase",
     "same meaning, different word order"),
    ("marie_best_friend", "bob_best_friend",
     "same predicate, different name (bob)"),
    ("marie_best_friend", "unrelated_weather",
     "topically unrelated"),
    ("anna_la", "anna_sf",
     "same predicate, different filler [core plurality case]"),
    ("anna_la", "anna_la_past",
     "same predicate+filler, different temporal framing"),
    ("marie_best_friend", "marie_sister",
     "same name, different relation"),
]

COLLAPSE_THRESHOLD = 0.98


# ── model loading ─────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


# ── embedding ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_sentence_embedding(text: str, method: str, tokenizer, model) -> torch.Tensor:
    """Return a (hidden_dim,) float32 embedding for *text*."""
    inputs = tokenizer(text, return_tensors="pt")
    # move every tensor to the same device as the model's first parameter
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple of (n_layers+1) tensors, each (1, seq_len, hidden)
    hidden_states = outputs.hidden_states

    attention_mask = inputs["attention_mask"]  # (1, seq_len)

    if method == "last_token":
        last_layer = hidden_states[-1]          # (1, seq_len, hidden)
        # find the last non-padding position
        seq_lengths = attention_mask.sum(dim=1) - 1  # (1,)
        vec = last_layer[0, seq_lengths[0], :]  # (hidden,)

    elif method == "mean":
        last_layer = hidden_states[-1]          # (1, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
        vec = (last_layer * mask).sum(dim=1) / mask.sum(dim=1)  # (1, hidden)
        vec = vec.squeeze(0)

    elif method == "mean_midlayer":
        mid = hidden_states[MID_LAYER]          # (1, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).float()
        vec = (mid * mask).sum(dim=1) / mask.sum(dim=1)
        vec = vec.squeeze(0)

    else:
        raise ValueError(f"Unknown pooling method: {method!r}")

    return vec.float().cpu()


# ── similarity matrix ─────────────────────────────────────────────────────────
def compute_similarity_matrix(embeddings: dict) -> dict:
    keys  = list(embeddings.keys())
    vecs  = torch.stack([embeddings[k] for k in keys])     # (N, hidden)
    norms = F.normalize(vecs, dim=1)
    sim   = (norms @ norms.T).numpy()                       # (N, N)
    return {"keys": keys, "matrix": sim}


# ── reporting helpers ─────────────────────────────────────────────────────────
def print_matrix(keys, matrix):
    col_w = max(len(k) for k in keys)
    header = f"{'':>{col_w}s}  " + "  ".join(f"{k:>8s}" for k in keys)
    print(header)
    for i, row_key in enumerate(keys):
        row = f"{row_key:>{col_w}s}  " + "  ".join(f"{matrix[i,j]:8.3f}" for j in range(len(keys)))
        print(row)


def save_matrix_csv(keys, matrix, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + keys)
        for i, k in enumerate(keys):
            writer.writerow([k] + [f"{matrix[i,j]:.6f}" for j in range(len(keys))])


def focal_report(keys, matrix, method) -> list[str]:
    idx = {k: i for i, k in enumerate(keys)}
    lines = [f"\n── Focal pairs ({method}) ──"]
    collapse_flags = []

    for a, b, label in FOCAL_PAIRS:
        sim = matrix[idx[a], idx[b]]
        flag = "  *** COLLAPSE ***" if sim > COLLAPSE_THRESHOLD else ""
        lines.append(f"  {a:35s} vs {b:35s}  {sim:.3f}  [{label}]{flag}")
        if flag:
            collapse_flags.append((a, b, sim, label))

    return lines, collapse_flags


def final_summary(all_focal: dict) -> list[str]:
    lines = ["\n" + "="*80, "FINAL SUMMARY", "="*80]

    for method, (focal_lines, collapses) in all_focal.items():
        lines.append(f"\n[{method}]")
        lines += focal_lines

        if collapses:
            lines.append(f"\n  !! {len(collapses)} pair(s) above collapse threshold {COLLAPSE_THRESHOLD}:")
            for a, b, sim, label in collapses:
                lines.append(f"     {a} vs {b}  sim={sim:.3f}  ({label})")

    # ── overall verdict ───────────────────────────────────────────────────────
    lines.append("\n" + "-"*80)
    lines.append("INTERPRETATION")
    lines.append("-"*80)

    total_collapses = sum(len(v[1]) for v in all_focal.values())

    lines.append(
        "Key question: does Qwen2.5-3B's representation space carry enough signal "
        "across name / predicate / filler variation that a retrieval head could "
        "plausibly learn to distinguish them?"
    )
    lines.append("")

    # collect anchor similarities for heuristic verdict
    verdicts = []
    for method, (_, _) in all_focal.items():
        pass  # rebuilt below from stored matrices

    # rebuilt from all_focal — we store the full focal lines, parse sims back out
    # (simpler: we'll assess in the run loop and pass through)
    if total_collapses == 0:
        lines.append(
            "No collapse detected across any pooling method. "
            "All focal pairs remain below the 0.98 threshold. "
            "There is meaningful geometric separation in this space — "
            "a retrieval head should be able to distinguish name, predicate, "
            "and filler variation at least at the embedding level."
        )
    else:
        lines.append(
            f"WARNING: {total_collapses} focal pair(s) exceeded the collapse threshold (>={COLLAPSE_THRESHOLD}). "
            "Those pairs are effectively indistinguishable to a dot-product retrieval head. "
            "Consider using a method that did NOT collapse, or fine-tuning the "
            "representation layer before building the memory store."
        )

    lines.append(
        "\nPooling guidance:\n"
        "  last_token   — causal LMs concentrate next-token prediction signal here;\n"
        "                 may conflate sentences that predict similar continuations.\n"
        "  mean         — averages token contributions; tends to smooth out fine-\n"
        "                 grained lexical differences but is more stable.\n"
        f"  mean_midlayer— layer {MID_LAYER} is deep enough to be contextual but\n"
        "                 shallower than output layers where generation artifacts\n"
        "                 accumulate; often the best tradeoff for retrieval tasks."
    )

    return lines


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer, model = load_model()

    probe_keys      = list(probes.keys())
    all_focal       = {}
    report_lines    = ["Embedding Noise-Floor Diagnostic Report", f"Model: {MODEL_ID}", ""]

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Pooling method: {method}")
        print(f"{'='*60}")

        embeddings = {
            k: get_sentence_embedding(text, method, tokenizer, model)
            for k, text in probes.items()
        }

        result  = compute_similarity_matrix(embeddings)
        keys    = result["keys"]
        matrix  = result["matrix"]

        print_matrix(keys, matrix)

        csv_path = os.path.join(OUT_DIR, f"similarity_{method}.csv")
        save_matrix_csv(keys, matrix, csv_path)
        print(f"\nSaved: {csv_path}")

        focal_lines, collapses = focal_report(keys, matrix, method)
        for line in focal_lines:
            print(line)

        all_focal[method] = (focal_lines, collapses)

        report_lines.append(f"Method: {method}")
        report_lines += focal_lines
        report_lines.append("")

    summary_lines = final_summary(all_focal)
    for line in summary_lines:
        print(line)

    report_lines += summary_lines

    report_path = os.path.join(OUT_DIR, "noise_floor_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
