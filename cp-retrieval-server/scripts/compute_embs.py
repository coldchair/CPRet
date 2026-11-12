import os
import argparse
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
import random
import json
import tqdm as tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        default='coldchair16/CPRetriever-Prob-Qwen3-4B-2510',
                        help="Path to the SentenceTransformer model")
    parser.add_argument("--input_path",
                        type=str,
                        default='./probs.jsonl',
                        help="Path to the input JSONL file containing sentences")
    parser.add_argument("--max_length",
                        type=int,
                        default=2048,
                        help="Maximum length of input sentences")

    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Optional provenance manifest (requires a local model path)")
    args = parser.parse_args()

    sentences = []

    input_path = args.input_path
    with open(input_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                data = json.loads(line)
                sentences.append(data['text'])

    print(f"Number of sentences: {len(sentences)}")

    save_dir = './'
    os.makedirs(save_dir, exist_ok=True)

    model_path = args.model_path
    model_name = os.path.basename(model_path.rstrip('/'))

    model = SentenceTransformer(model_path, trust_remote_code=True)
    model.tokenizer.model_max_length = args.max_length
    model.max_seq_length = args.max_length

    pool = model.start_multi_process_pool()
    emb = model.encode_multi_process(sentences, pool, show_progress_bar=True, batch_size=8)
    model.stop_multi_process_pool(pool)

    print("Embeddings computed. Shape:", emb.shape)

    save_name = f"{input_path.split('/')[-1].replace('.jsonl', '')}_embs.npy"
    save_path = os.path.join(save_dir, save_name)
    emb = emb.astype('float32')
    np.save(save_path, emb)

    print(f"Embeddings saved to {save_path}")
    if args.manifest_path:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from retrieval.retrieval_state import model_fingerprint, sha256_file

        manifest = {
            "schema_version": 1,
            "model_fingerprint": model_fingerprint(model_path),
            "max_seq_length": args.max_length,
            "count": len(sentences),
            "dimension": emb.shape[1],
            "sha256": {
                "problems.jsonl": sha256_file(Path(input_path)),
                "embeddings.npy": sha256_file(Path(save_path)),
            },
        }
        Path(args.manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
