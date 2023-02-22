from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ao3_disco_ranking.ao3 import get_work_jsons
from ao3_disco_ranking.types import WorkID


class EmbeddingRanker:
    def __init__(self, model, works):
        self.model = model

        self.work_to_idx = {work: i for i, work in enumerate(works)}
        self.idx_to_work = {k: v for v, k in self.work_to_idx.items()}

        embeddings = np.zeros((len(self.work_to_idx), model.output_dim))
        for work, i in tqdm(self.work_to_idx.items()):
            embeddings[i] = model.embedding(work, works)
        self.embeddings = torch.tensor(embeddings)

    def rank(
        self, work_id: WorkID, candidates: Optional[Iterable[WorkID]] = None, num_results: int = 50
    ) -> List[Tuple[WorkID, float]]:
        if candidates:
            valid_idx = [self.work_to_idx[workID] for workID in candidates]
            embeddings = self.embeddings[valid_idx]
            idx_to_work = {i: workID for i, workID in enumerate(candidates)}
        else:
            embeddings = self.embeddings
            idx_to_work = self.idx_to_work

        if work_id in self.work_to_idx:
            vec = self.embeddings[self.work_to_idx[work_id]]
        else:
            work_json = list(get_work_jsons([work_id]))[0]
            works = {work_id: work_json}
            vec = self.model.embedding(work_id, works)
        scores = F.cosine_similarity(vec, embeddings)

        results = []
        values, indices = torch.topk(scores, min(num_results, len(scores)))
        for v, i in zip(values, indices):
            other_id = idx_to_work[i.item()]
            if other_id == work_id:
                continue
            results.append((other_id, v.item()))
        return results
