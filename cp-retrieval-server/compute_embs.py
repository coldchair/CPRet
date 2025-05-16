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
                        default='coldchair16/CPRetriever-Prob',
                        help="Path to the SentenceTransformer model")

    args = parser.parse_args()

    sentences = []

    input_path = './probs.jsonl'
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
    model.tokenizer.model_max_length = 1024
    model.max_seq_length = 1024

    pool = model.start_multi_process_pool()
    emb = model.encode_multi_process(sentences, pool, show_progress_bar=True, batch_size=8)
    model.stop_multi_process_pool(pool)

    print("Embeddings computed. Shape:", emb.shape)

    save_path = os.path.join(save_dir, f"probs_embs.npy")
    emb = emb.astype('float32')
    np.save(save_path, emb)

    print(f"Embeddings saved to {save_path}")
