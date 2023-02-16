"""First stage models.

These models are designed to be used to generate an initial embedding pool using an
approximate nearest neighbors library.
"""
import json
import pickle
from random import shuffle
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from .base import BaseModel, WorkID


class FeatureExtractor:
    def fit(self, works: Dict[str, Any]):
        self.dense_dims = 12  # hits, kudos, words, comments, bookmarks
        self.sparse_features = {}

        # Normal sparse features
        for tag_type in [
            "fandom",
            "rating",
            "category",
            "relationship",
            "character",
            "freeform",
        ]:
            tag_to_i: Dict[str, int] = {}
            for work in works.values():
                for tag in work["tags"][tag_type]:
                    if tag not in tag_to_i:
                        tag_to_i[tag] = len(tag_to_i)
            self.sparse_features[tag_type] = tag_to_i

        X = np.stack([self._raw_transform(work)[0] for work in works.values()])
        self.scaler = RobustScaler()
        self.scaler.fit(X)

    def transform(self, work: Any):
        dense, sparse = self._raw_transform(work)
        return self.scaler.transform(dense.reshape(1, -1))[0], sparse

    def _raw_transform(self, work: Any):
        dense = np.zeros(self.dense_dims)
        for i, tag in enumerate(["hits", "kudos", "words", "comments", "bookmarks"]):
            try:
                dense[i] = int(work["statistics"][tag])
            except KeyError:
                pass
            except ValueError:
                pass
        try:
            dense[5] = "/" in work["chapters"]
            dense[6] = int(work["chapters"].split("/")[0])
            dense[7] = int(work["chapters"].split("/")[1])
            dense[8] = dense[6] == dense[7]  # Is it complete?
        except KeyError:
            pass
        except ValueError:
            pass
        try:
            dense[9] = int(work["status"].split("-")[0])
            dense[10] = int(work["published"].split("-")[0])
        except KeyError:
            pass
        except ValueError:
            pass
        dense[11] = len(work["summary"])

        sparse: Dict[str, List[int]] = {}
        for tag_type in [
            "fandom",
            "rating",
            "category",
            "relationship",
            "character",
            "freeform",
        ]:
            sparse[tag_type] = []
            for tag in work["tags"][tag_type]:
                if tag not in self.sparse_features[tag_type]:
                    continue
                sparse[tag_type].append(self.sparse_features[tag_type][tag])

        return dense, sparse


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
        return z


class FirstModel(BaseModel):
    def __init__(
        self,
        lr=1e-2,
        num_epochs=2,
        batch_size=100,
        output_dim=128,
        max_hash_size=50000,
        dropout=0.0,
        use_batch_norm=False,
    ):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.max_hash_size = max_hash_size
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        with open("data/works_collections.pkl", "rb") as fin:
            works, _ = pickle.load(fin)
            self.works = works

    def fit(self, path_to_jsonl):
        dataset = []
        work_to_json = {}
        with open(path_to_jsonl, "rt") as fin:
            for x in tqdm(map(json.loads, fin)):
                dataset.append(x)
                work_to_json[x["work"]] = self.works[x["work"]]
                for work, _ in x["candidates"].items():
                    work_to_json[work] = self.works[work]

        self.featurizer = FeatureExtractor()
        self.featurizer.fit(work_to_json)
        work_to_features = {k: self.featurizer.transform(v) for k, v in work_to_json.items()}

        self.model = EmbeddingModule(
            self.featurizer,
            self.output_dim,
            self.max_hash_size,
            self.dropout,
            self.use_batch_norm,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            shuffle(dataset)

            def _batched_dataset(dataset, batch_size=self.batch_size):
                n_samples = len(dataset)
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    yield dataset[start:end]

            losses = []
            iterator = tqdm(_batched_dataset(dataset))
            for batch in iterator:
                works, indices, y_true = [], [], []
                for row in batch:
                    candidates, scores = zip(*row["candidates"].items())

                    anchor_idx = len(works)
                    candidate_idx = (len(works) + 1, len(works) + len(candidates) + 1)
                    indices.append((anchor_idx, candidate_idx))

                    works.append(row["work"])
                    works.extend(candidates)

                    y_true.append(scores)

                embeddings = self.model(works, work_to_features)

                loss = 0.0
                for (anchor_idx, (start_idx, end_idx)), y in zip(indices, y_true):
                    y_pred = 10.0 * F.cosine_similarity(
                        embeddings[anchor_idx], embeddings[start_idx:end_idx]
                    )
                    y_true = F.softmax(torch.FloatTensor(y), dim=0)
                    y_pred = F.softmax(y_pred, dim=0)
                    loss += -torch.sum(y_true * torch.log(y_pred))
                loss /= len(indices)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                iterator.set_description_str(f"Epoch {epoch} | Loss {losses[-1]:.4f}")
            print(f"epoch: {epoch}")
            print(f"loss: {np.mean(losses)}")

        self.model.eval()

    def embedding(self, work_id: WorkID) -> np.ndarray:
        work_features = {work_id: self.featurizer.transform(self.works[work_id])}
        with torch.no_grad():
            return self.model([work_id], work_features).detach().numpy()[0]

    def rank(self, work_id, candidates):
        all_works = [work_id] + list(candidates)
        work_features = {k: self.featurizer.transform(self.works[k]) for k in all_works}
        with torch.no_grad():
            embeddings = self.model(all_works, work_features)
        scores = F.cosine_similarity(embeddings[0], embeddings[1:])
        return list(zip(list(candidates), scores.detach().numpy()))
