"""First stage models.

These models are designed to be used to generate an initial embedding pool using an
approximate nearest neighbors library.
"""
import json
import pickle
from random import shuffle
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseModel, WorkID
from .embedding import EmbeddingModule
from .features import FeatureExtractor


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

    def fit(self, path_to_jsonl):
        with open("data/works_collections.pkl", "rb") as fin:
            works, _ = pickle.load(fin)

        dataset = []
        work_to_json = {}
        with open(path_to_jsonl, "rt") as fin:
            for x in tqdm(map(json.loads, fin)):
                dataset.append(x)
                work_to_json[x["work"]] = works[x["work"]]
                for work, _ in x["candidates"].items():
                    work_to_json[work] = works[work]

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

                embeddings = self.model(works, work_to_features)["embedding"]

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

    def embedding(self, work_id: WorkID, works: Dict[WorkID, Any]) -> np.ndarray:
        work_features = {work_id: self.featurizer.transform(works[work_id])}
        with torch.no_grad():
            return self.model([work_id], work_features)["embedding"].detach()[0]

    def rank(self, work_id, candidates, works: Dict[WorkID, Any]):
        all_works = [work_id] + list(candidates)
        work_features = {k: self.featurizer.transform(works[k]) for k in all_works}
        with torch.no_grad():
            embeddings = self.model(all_works, work_features)["embedding"]
        scores = F.cosine_similarity(embeddings[0], embeddings[1:])
        return list(zip(list(candidates), scores.detach().numpy()))
