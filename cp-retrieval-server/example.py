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

    model_path = args.model_path
    model = SentenceTransformer(model_path, trust_remote_code=True)
    model.tokenizer.model_max_length = 1024
    model.max_seq_length = 1024

    embs = np.load('./probs_embs.npy')
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    probs_path = './probs.jsonl'
    probs = []
    with open(probs_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                data = json.loads(line)
                probs.append(data)
    
    
    text = '''
You are given a sequence that supports the following operations:
1. Flatten a specified range.
2. Calculate the sum of the counts of distinct numbers in all subsegments of
length k.'''

    text_emb = model.encode(text, convert_to_tensor=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    sim_mat = np.dot(embs, text_emb)
    print(sim_mat.shape)
    rank = np.argsort(sim_mat, axis=0)[::-1]

    p = probs[rank[0]]
    print(p['title'], p['url'], p['source'], p['text'])


    