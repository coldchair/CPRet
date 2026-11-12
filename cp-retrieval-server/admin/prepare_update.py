from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from retrieval.retrieval_state import (load_bundle, load_problem_data, model_fingerprint,
                             problem_key, sha256_file, text_key, validate_stable_prefix)


APP_DIR = Path(__file__).resolve().parents[1]


def stable_order(old: list[dict], incoming: list[dict]) -> list[int]:
    remaining = defaultdict(deque)
    for index, problem in enumerate(incoming):
        remaining[problem_key(problem)].append(index)
    order = []
    for pid, problem in enumerate(old):
        matches = remaining[problem_key(problem)]
        if not matches:
            raise ValueError(f"Old problem #{pid} is missing; deletion would break existing /p/<pid> links")
        exact = next((i for i in matches if text_key(incoming[i]["text"]) == text_key(problem["text"])), None)
        index = exact if exact is not None else matches[0]
        matches.remove(index)
        order.append(index)
    used = set(order)
    order.extend(index for index in range(len(incoming)) if index not in used)
    return order


def prepare_bundle(input_path: Path, output_dir: Path, base, fingerprint: str,
                   target_length: int, token_count, encode_one=None, precomputed=None,
                   progress=None, cancelled=None) -> dict:
    if target_length not in (2048, 8192):
        raise ValueError("Bundle length must be 2048 or 8192")
    if output_dir.exists():
        raise ValueError("Output bundle already exists; use a new version directory")
    with input_path.open(encoding="utf-8") as stream:
        incoming = [json.loads(line) for line in stream]
    for row in incoming:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("Incoming problems require string text")
    order = stable_order(base.probs, incoming)
    records = [incoming[index] for index in order]
    validate_stable_prefix(base.probs, records)
    base_length = base.manifest["max_seq_length"] if base.manifest else 2048
    reusable = {}
    for index, row in enumerate(base.probs):
        text = row["text"]
        if not text.strip():
            continue
        if base_length == target_length or token_count(text) <= min(base_length, target_length):
            reusable.setdefault(text_key(text), index)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=output_dir.parent))
    dimension = base.embeddings.shape[1]
    counters = {"reused": 0, "computed_unique": 0, "deduplicated": 0, "precomputed": 0}
    try:
        vectors = np.lib.format.open_memmap(staging / "embeddings.npy", mode="w+",
                                            dtype=np.float32, shape=(len(records), dimension))
        completed = {}
        for index, row in enumerate(records):
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Bundle build cancelled")
            key = text_key(row["text"])
            if not row["text"].strip():
                vector = base.embeddings[index]
                counters["reused"] += 1
            elif key in completed:
                vector = vectors[completed[key]]
                counters["deduplicated"] += 1
            elif precomputed is not None and key in precomputed:
                vector = precomputed[key]
                counters["precomputed"] += 1
            elif key in reusable:
                vector = base.embeddings[reusable[key]]
                counters["reused"] += 1
            else:
                if encode_one is None:
                    raise ValueError(f"Missing compatible embedding for row {index}; compute this text at {target_length} tokens")
                vector = encode_one(row["text"], target_length)
                counters["computed_unique"] += 1
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (dimension,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 0:
                raise ValueError(f"Invalid generated embedding at row {index}")
            vectors[index] = vector / np.linalg.norm(vector)
            if row["text"].strip():
                completed[key] = index
            if progress is not None and (index % 100 == 0 or index + 1 == len(records)):
                progress({"done": index + 1, "total": len(records), **counters})
        vectors.flush()
        del vectors
        with (staging / "problems.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest = {"schema_version": 1, "model_fingerprint": fingerprint,
                    "max_seq_length": target_length, "dimension": dimension, "count": len(records),
                    "stats": counters,
                    "preserved_empty_rows": [i for i, p in enumerate(records) if not p["text"].strip()],
                    "sha256": {name: sha256_file(staging / name) for name in ("problems.jsonl", "embeddings.npy")}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        load_bundle(staging, fingerprint, dimension)
        if output_dir.exists():
            raise ValueError("Output bundle appeared during build; refusing overwrite")
        staging.rename(output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable incremental embedding bundle, retaining old numeric problem IDs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=APP_DIR / "CPRetriever-Prob-Qwen3-4B-2510")
    parser.add_argument("--base-bundle", type=Path)
    parser.add_argument("--base-probs", type=Path, default=APP_DIR / "probs_2611.jsonl")
    parser.add_argument("--base-embs", type=Path, default=APP_DIR / "probs_2611_embs.npy")
    parser.add_argument("--trust-legacy-embeddings", action="store_true")
    parser.add_argument("--max-length", type=int, choices=(2048, 8192), default=8192)
    parser.add_argument("--backend", choices=("vllm", "sentence_transformers"), default="vllm")
    parser.add_argument("--import-probs", type=Path)
    parser.add_argument("--import-embs", type=Path)
    parser.add_argument("--import-manifest", type=Path)
    parser.add_argument("--compute-missing", action="store_true")
    args = parser.parse_args()
    fingerprint = model_fingerprint(str(args.model_path))
    from transformers import AutoConfig, AutoTokenizer
    dimension = AutoConfig.from_pretrained(args.model_path).hidden_size
    if args.base_bundle:
        base = load_bundle(args.base_bundle, fingerprint, dimension)
    else:
        if not args.trust_legacy_embeddings:
            parser.error("Legacy vectors have no model provenance: explicitly pass --trust-legacy-embeddings")
        base = load_problem_data(args.base_probs, args.base_embs, expected_dimension=dimension,
                                 allow_empty_text=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.model_max_length = 10000000

    def count_tokens(text):
        return len(tokenizer(text.strip(), add_special_tokens=True, truncation=False)["input_ids"])

    imported = None
    if any((args.import_probs, args.import_embs, args.import_manifest)):
        if not all((args.import_probs, args.import_embs, args.import_manifest)):
            parser.error("Import needs aligned --import-probs, --import-embs and --import-manifest")
        provenance = json.loads(args.import_manifest.read_text(encoding="utf-8"))
        if provenance.get("model_fingerprint") != fingerprint or provenance.get("max_seq_length") != args.max_length:
            parser.error("Imported embeddings have incompatible model/length provenance")
        if provenance.get("sha256", {}).get("problems.jsonl") != sha256_file(args.import_probs) or provenance.get("sha256", {}).get("embeddings.npy") != sha256_file(args.import_embs):
            parser.error("Imported vectors fail provenance checksums")
        data = load_problem_data(args.import_probs, args.import_embs, expected_dimension=dimension)
        imported = {text_key(p["text"]): data.embeddings[i] for i, p in enumerate(data.probs)}

    encoder = None

    def encode_one(text, length):
        nonlocal encoder
        if encoder is None:
            from encoder.query_encoder import create_encoder
            encoder = create_encoder(str(args.model_path), backend=args.backend, max_seq_length=length)
        return encoder.encode(text)

    manifest = prepare_bundle(args.input, args.output, base, fingerprint, args.max_length, count_tokens,
                              encode_one if args.compute_missing else None, imported,
                              progress=lambda p: print(json.dumps(p), flush=True))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
