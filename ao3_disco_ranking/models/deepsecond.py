import json
import pickle
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseModel, WorkID
from .embedding import EmbeddingModule
from .features import FeatureExtractor


class PairwiseRankingModule(nn.Module):
    def __init__(
        self,
        fe,
        output_dim=128,
        max_hash_size=50000,
        dropout=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.embedding = EmbeddingModule(
            fe,
            output_dim=output_dim,
            max_hash_size=max_hash_size,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.linear = nn.Sequential(nn.BatchNorm1d(7), nn.Linear(7, 1))

    def forward(self, work_pairs, work_features):
        x1, x2 = zip(*work_pairs)
        x1 = self.embedding(x1, work_features)
        x2 = self.embedding(x2, work_features)
        interactions = torch.stack(
            [
                torch.cosine_similarity(x1["embedding"], x2["embedding"], dim=1),
                torch.cosine_similarity(x1["dense_embedding"], x2["dense_embedding"], dim=1),
                torch.cosine_similarity(
                    x1["sparse_embeddings"]["fandom"], x2["sparse_embeddings"]["fandom"], dim=1
                ),
                torch.cosine_similarity(
                    x1["sparse_embeddings"]["category"], x2["sparse_embeddings"]["category"], dim=1
                ),
                torch.cosine_similarity(
                    x1["sparse_embeddings"]["relationship"],
                    x2["sparse_embeddings"]["relationship"],
                    dim=1,
                ),
                torch.cosine_similarity(
                    x1["sparse_embeddings"]["character"],
                    x2["sparse_embeddings"]["character"],
                    dim=1,
                ),
                torch.cosine_similarity(
                    x1["sparse_embeddings"]["freeform"], x2["sparse_embeddings"]["freeform"], dim=1
                ),
            ],
            dim=1,
        )
        assert not torch.isnan(interactions).any()
        return {
            "score": self.linear(interactions)[:, 0],
            "embedding1": x1["embedding"],
            "embedding2": x2["embedding"],
        }


class DeepSecondModel(BaseModel):
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

        self.model = PairwiseRankingModule(
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
                work_pairs, indices, y_true = [], [], []
                for row in batch:
                    candidates, scores = zip(*row["candidates"].items())

                    start_idx = len(work_pairs)
                    end_idx = len(work_pairs) + len(candidates)
                    indices.append((start_idx, end_idx))

                    for c in candidates:
                        work_pairs.append((row["work"], c))

                    y_true.append(scores)

                results = self.model(work_pairs, work_to_features)

                loss = 0.0
                for (start_idx, end_idx), y in zip(indices, y_true):
                    y_pred = results["score"][start_idx:end_idx]
                    y_true = F.softmax(torch.FloatTensor(y), dim=0)
                    y_pred = F.softmax(y_pred, dim=0)
                    loss += -torch.sum(y_true * torch.log(y_pred))
                loss /= len(indices) * 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                iterator.set_description_str(f"Epoch {epoch} | Loss {losses[-1]:.4f}")
            print(f"epoch: {epoch}")
            print(f"loss: {np.mean(losses)}")

        self.model.eval()

    def rank(self, work_id, candidates):
        work_pairs = [(work_id, c) for c in list(candidates)]
        work_features = {
            k: self.featurizer.transform(self.works[k]) for k in [work_id] + list(candidates)
        }
        with torch.no_grad():
            scores = self.model(work_pairs, work_features)["score"]
        return list(zip(list(candidates), scores.detach().numpy()))
