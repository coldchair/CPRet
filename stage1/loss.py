import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

class InfoNCELoss_gradcache(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss_gradcache, self).__init__()
        self.temperature = temperature

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, **loss_kwargs) -> torch.Tensor:
        # Convert bf16 to fp32 for numerical stability
        query_embeddings = query_embeddings.float()
        doc_embeddings = doc_embeddings.float()

        # Gather embeddings across distributed processes
        query_embeddings = self.gather_tensor(query_embeddings)
        doc_embeddings = self.gather_tensor(doc_embeddings)

        # Normalize the embeddings to unit vectors
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

        return self._compute_loss(query_embeddings, doc_embeddings)

    def _compute_loss(self, query_embeddings, doc_embeddings):
        # Positive similarity (dot product between aligned query-doc pairs)
        sim_u = torch.sum(query_embeddings * doc_embeddings, dim=1) / self.temperature
        sim_u = torch.exp(sim_u)

        # Similarities with all queries and docs (positive + negatives)
        sim_v = torch.sum(torch.exp(torch.matmul(query_embeddings, doc_embeddings.T) / self.temperature), dim=1) + \
                torch.sum(torch.exp(torch.matmul(query_embeddings, query_embeddings.T) / self.temperature), dim=1)
        # Remove self-similarity term
        sim_v = sim_v - torch.exp(torch.tensor(1 / self.temperature, device=sim_v.device))

        return -torch.log(sim_u / sim_v).mean()
    
    def gather_tensor(self, t):
        # All-gather operation for distributed training
        t_new = torch.distributed.nn.all_gather(t)
        t_new = torch.cat(t_new, dim=0)
        return t_new


# Version that uses sum of positive similarities in numerator
class InfoNCELoss_gradcache_multipos(InfoNCELoss_gradcache):
    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, **loss_kwargs) -> torch.Tensor:
        query_embeddings = query_embeddings.float()
        doc_embeddings = doc_embeddings.float()

        query_embeddings = self.gather_tensor(query_embeddings)
        doc_embeddings = self.gather_tensor(doc_embeddings)

        # Reshape to [batch_size, num_pos, dim]
        doc_embeddings = doc_embeddings.reshape(query_embeddings.shape[0], -1, query_embeddings.shape[1])

        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=2)

        sim_mat = torch.exp(query_embeddings @ query_embeddings.T / self.temperature)
        sim_v = torch.sum(sim_mat, dim=1) - torch.exp(torch.tensor(float(1.0) / self.temperature, device=sim_mat.device))

        sim_w = torch.einsum("ik, jnk -> ijn", query_embeddings, doc_embeddings) / self.temperature
        sim_w = torch.exp(sim_w)  # shape: [bs, bs, pos]
        sim_w = torch.sum(sim_w, dim=2)  # shape: [bs, bs]
        sim_u = sim_w.diagonal()  # positive similarities on the diagonal
        sim_w = torch.sum(sim_w, dim=1)  # total similarities

        return -torch.log(sim_u / (sim_w + sim_v)).mean() 


# Version that separates query-doc and query-query terms
class InfoNCELoss_gradcache_multipos(InfoNCELoss_gradcache):
    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, **loss_kwargs) -> torch.Tensor:
        query_embeddings = query_embeddings.float()
        doc_embeddings = doc_embeddings.float()

        query_embeddings = self.gather_tensor(query_embeddings)
        doc_embeddings = self.gather_tensor(doc_embeddings)

        doc_embeddings = doc_embeddings.reshape(query_embeddings.shape[0], -1, query_embeddings.shape[1])

        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=2)

        sim_mat = torch.exp(query_embeddings @ query_embeddings.T / self.temperature)
        sim_v = torch.sum(sim_mat, dim=1) - torch.exp(torch.tensor(float(1.0) / self.temperature, device=sim_mat.device))

        # Positive similarities: shape [bs, pos]
        sim_a = torch.exp(torch.sum(query_embeddings.unsqueeze(1) * doc_embeddings, dim=2) / self.temperature)

        sim_w = torch.einsum("ik, jnk -> ijn", query_embeddings, doc_embeddings) / self.temperature
        sim_w = torch.exp(sim_w)  # shape: [bs, bs, pos]

        sim_p = torch.sum(sim_w, dim=2)  # [bs, bs]
        sim_u_origin = sim_p.diagonal()
        sim_q = torch.sum(sim_p, dim=1)

        return torch.mean(-torch.log(sim_a / ((sim_q + sim_v - sim_u_origin).unsqueeze(1) + sim_a)))


# Version that implements Group-InfoNCE with regularization
class GroupInfoNCELoss_gradcache_multipos(InfoNCELoss_gradcache):
    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, **loss_kwargs) -> torch.Tensor:
        query_embeddings = query_embeddings.float()
        doc_embeddings = doc_embeddings.float()

        query_embeddings = self.gather_tensor(query_embeddings)
        doc_embeddings = self.gather_tensor(doc_embeddings)

        doc_embeddings = doc_embeddings.reshape(query_embeddings.shape[0], -1, query_embeddings.shape[1])

        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=2)

        # sim_a: similarity between each query and its group of positive docs
        sim_a = torch.einsum('ik, ijk -> ij', query_embeddings, doc_embeddings)
        loss_penalty = torch.var(sim_a, dim=1).mean()  # penalty: variance within group

        sim_b = torch.mean(sim_a, dim=1)
        sim_b = torch.exp(sim_b / self.temperature)

        # query-to-query similarity matrix (excluding self-similarity)
        sim_p = torch.einsum('ik, jk -> ij', query_embeddings, query_embeddings)
        sim_p = torch.exp(sim_p / self.temperature)
        sim_p = torch.sum(sim_p, dim=1) - torch.exp(torch.tensor(1 / self.temperature, device=sim_p.device))

        # query-to-all-doc-groups similarity
        sim_q = torch.einsum('ik, jpk -> ijp', query_embeddings, doc_embeddings)
        sim_q = torch.mean(sim_q, dim=2)  # [bs, bs]
        sim_q = torch.exp(sim_q / self.temperature)
        sim_q = torch.sum(sim_q, dim=1)

        loss = -torch.log(sim_b / (sim_p + sim_q)).mean()

        if torch.distributed.get_rank() == 0:
            print(f"\nloss_penalty: {loss_penalty} / T^2 = {loss_penalty / self.temperature ** 2}")
            print(f"loss: {loss}")

        return loss + loss_penalty / self.temperature ** 2
