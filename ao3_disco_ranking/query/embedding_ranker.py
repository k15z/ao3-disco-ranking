from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ao3_disco_ranking.types import WorkID


class EmbeddingRanker:
    def __init__(self, model, works):
        self.model = model

        self.work_to_idx = {work: i for i, work in enumerate(works)}
        self.idx_to_work = {k: v for v, k in self.work_to_idx.items()}

        embeddings = np.zeros((len(self.work_to_idx), 128))
        for work, i in tqdm(self.work_to_idx.items()):
            embeddings[i] = model.embedding(work)
        self.embeddings = torch.tensor(embeddings)

    def rank(self, work_id: WorkID, candidates: List[WorkID]) -> List[Tuple[WorkID, float]]:
        if candidates:
            valid_idx = [self.work_to_idx[workID] for workID in candidates]
            embeddings = self.embeddings[valid_idx]
            idx_to_work = {i: workID for i, workID in enumerate(candidates)}
        else:
            embeddings = self.embeddings
            idx_to_work = self.idx_to_work

        if work_id in self.work_to_idx:
            idx = self.work_to_idx[work_id]
            vec = self.embeddings[idx]
        else:
            raise NotImplementedError("scrape work and embed?")
        scores = F.cosine_similarity(vec, embeddings)

        results = []
        values, indices = torch.topk(scores, 20)
        for v, i in zip(values, indices):
            results.append((idx_to_work[i.item()], v))
        return results
