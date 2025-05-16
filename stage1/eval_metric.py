import heapq
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from typing import TYPE_CHECKING, Callable
from torch import Tensor
from tqdm import trange

# Modified InformationRetrievalEvaluator to support directly passing in embeddings
class CustomRetrievalEvaluator(InformationRetrievalEvaluator):
    def __init__(
        self,
        corpus_chunk_size: int = 50000,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10, 100],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [10, 100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = {
            "cos": SimilarityFunction.to_similarity_fn("cosine")
        },
        main_score_function: str | SimilarityFunction | None = None,
    ) -> None:
        # Manually call base class __init__ (bypassing argument mismatches)
        super(InformationRetrievalEvaluator).__init__()
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self._append_csv_headers(self.score_function_names)

    def calc_results(self, query_embeddings, doc_embeddings, r_docs):
        # Construct internal ID mappings and relevance ground truth
        self.queries_ids = [f"q{i}" for i in range(len(query_embeddings))]
        self.corpus_ids = [f"c{i}" for i in range(len(doc_embeddings))]
        self.relevant_docs = {
            f"q{i}": set([f"c{j}" for j in r_docs[i]]) for i in range(len(r_docs))
        }
        self.queries = [""] * len(query_embeddings)
        self.corpus = [""] * len(doc_embeddings)

        # Determine the largest k needed
        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Initialize result storage
        queries_result_list = {
            name: [[] for _ in range(len(query_embeddings))]
            for name in self.score_functions
        }

        # Process corpus in chunks to avoid memory overflow
        for corpus_start_idx in trange(
            0, len(self.corpus), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))
            sub_corpus_embeddings = doc_embeddings[corpus_start_idx:corpus_end_idx]

            for name, score_function in self.score_functions.items():
                # Compute similarity scores
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Top-k scores and indices
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        # Do not skip query==corpus as some setups use identical indexing
                        if len(queries_result_list[name][query_itr]) < max_k:
                            heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))
                        else:
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))

        # Convert heap tuples to result dicts
        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                queries_result_list[name][query_itr] = [
                    {"corpus_id": corpus_id, "score": score}
                    for score, corpus_id in queries_result_list[name][query_itr]
                ]

        # Compute standard IR metrics
        scores = {
            name: self.compute_metrics(queries_result_list[name])
            for name in self.score_functions
        }

        # Compute average cosine similarity between each query and its relevant docs
        for name, score_func in self.score_functions.items():
            v_list = []
            for i in range(len(query_embeddings)):
                score_v = score_func(query_embeddings[i].reshape(1, -1), doc_embeddings[r_docs[i]])
                v_list.append(score_v.mean().item())
            v_list = np.array(v_list)
            scores["cos"][f"{name}_mean"] = np.mean(v_list)
            p_list = [10, 50, 90]
            scores["cos"][f"{name}_percentile@k"] = {p: np.percentile(v_list, p) for p in p_list}

        return scores


def compute_metrics_custom(p):
    # p is an EvalPrediction object containing predictions and label_ids
    query_embeddings, doc_embeddings, ids, doc_ids = p.predictions

    n = len(ids)

    doc_embeddings_flatten = []
    rdocs = []
    tot = 0
    for i in range(n):
        num = doc_ids[i]  # number of documents for query i
        rdocs.append(list(range(tot, tot + num)))  # relevant doc indices for query i
        for j in range(num):
            doc_embeddings_flatten.append(doc_embeddings[i][j])
        tot += num

    doc_embeddings_flatten = np.array(doc_embeddings_flatten)

    evaluator = CustomRetrievalEvaluator()
    results = evaluator.calc_results(query_embeddings, doc_embeddings_flatten, rdocs)["cos"]

    # Flatten nested results into a flat dictionary
    new_results = {}
    for k in results.keys():
        if isinstance(results[k], dict):
            for kk in results[k].keys():
                new_results[f"{k}_{kk}"] = results[k][kk]
        else:
            new_results[k] = results[k]

    # Log results only on rank 0
    if torch.distributed.get_rank() == 0:
        print(results)

    return new_results
