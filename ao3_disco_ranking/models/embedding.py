"""The PyTorch module which forms the backbone of all the deep ranking models."""
import numpy as np
import torch
import torch.nn as nn


class EmbeddingModule(nn.Module):
    def __init__(
        self,
        fe,
        output_dim=128,
        max_hash_size=50000,
        dropout=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        self.max_hash_size = max_hash_size
        self.embedding_dims = {
            "rating": 10,
            "category": 10,
            "fandom": 20,
            "relationship": 20,
            "character": 20,
            "freeform": 20,
        }

        self.embeddings = {}
        for k, v in self.embedding_dims.items():
            self.embeddings[k] = nn.EmbeddingBag(
                min(len(fe.sparse_features[k]), max_hash_size),
                v,
                max_norm=1.0,
                mode="sum",
            )
        self.embeddings = nn.ModuleDict(self.embeddings)

        self.dense_arch = nn.Linear(fe.dense_dims, 20)

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(20 + 100 + 15)
        self.over_arch = nn.Sequential(
            nn.Linear(20 + 100 + 15, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x, work_features):
        dense, sparse = zip(*(work_features[work] for work in x))

        dense_embedding = self.dense_arch(torch.FloatTensor(np.array(dense)))

        sparse_embeddings = {}
        for tag in self.embedding_dims:
            id_list, offsets = [], []
            for sparse_row in sparse:
                offsets.append(len(id_list))
                id_list.extend([x % self.max_hash_size for x in sparse_row[tag]])
            id_list, offsets = torch.LongTensor(id_list), torch.LongTensor(offsets)
            sparse_embeddings[tag] = self.embeddings[tag](id_list, offsets)

        sparse_embedding_basic = torch.cat(
            [
                sparse_embeddings["rating"],
                sparse_embeddings["category"],
            ],
            dim=-1,
        )
        pre_over = torch.cat(
            [dense_embedding]
            + [
                sparse_embeddings["rating"],
                sparse_embeddings["category"],
                sparse_embeddings["fandom"],
                sparse_embeddings["relationship"],
                sparse_embeddings["character"],
                sparse_embeddings["freeform"],
            ]
            + [
                torch.sum(dense_embedding * sparse_embedding_basic, dim=-1).unsqueeze(-1),
                torch.sum(dense_embedding * sparse_embeddings["fandom"], dim=-1).unsqueeze(-1),
                torch.sum(dense_embedding * sparse_embeddings["relationship"], dim=-1).unsqueeze(
                    -1
                ),
                torch.sum(dense_embedding * sparse_embeddings["character"], dim=-1).unsqueeze(-1),
                torch.sum(dense_embedding * sparse_embeddings["freeform"], dim=-1).unsqueeze(-1),
                torch.sum(sparse_embedding_basic * sparse_embeddings["fandom"], dim=-1).unsqueeze(
                    -1
                ),
                torch.sum(
                    sparse_embedding_basic * sparse_embeddings["relationship"], dim=-1
                ).unsqueeze(-1),
                torch.sum(
                    sparse_embedding_basic * sparse_embeddings["character"], dim=-1
                ).unsqueeze(-1),
                torch.sum(sparse_embedding_basic * sparse_embeddings["freeform"], dim=-1).unsqueeze(
                    -1
                ),
                torch.sum(
                    sparse_embeddings["fandom"] * sparse_embeddings["relationship"],
                    dim=-1,
                ).unsqueeze(-1),
                torch.sum(
                    sparse_embeddings["fandom"] * sparse_embeddings["character"], dim=-1
                ).unsqueeze(-1),
                torch.sum(
                    sparse_embeddings["fandom"] * sparse_embeddings["freeform"], dim=-1
                ).unsqueeze(-1),
                torch.sum(
                    sparse_embeddings["relationship"] * sparse_embeddings["character"],
                    dim=-1,
                ).unsqueeze(-1),
                torch.sum(
                    sparse_embeddings["relationship"] * sparse_embeddings["freeform"],
                    dim=-1,
                ).unsqueeze(-1),
                torch.sum(
                    sparse_embeddings["character"] * sparse_embeddings["freeform"],
                    dim=-1,
                ).unsqueeze(-1),
            ],
            dim=1,
        )
        if self.use_batch_norm:
            pre_over = self.batch_norm(pre_over)
        pre_over = self.dropout(pre_over)
        z = self.over_arch(pre_over)
        return {
            "embedding": z,
            "pre_over": pre_over,
            "dense_embedding": dense_embedding,
            "sparse_embeddings": sparse_embeddings,
        }
