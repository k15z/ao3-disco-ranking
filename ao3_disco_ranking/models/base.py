from abc import ABC
from random import shuffle
from typing import List, Tuple

import numpy as np

WorkID = str


class BaseModel(ABC):
    def fit(self, path_to_jsonl: str) -> None:
        """Train a ranking model."""
        pass

    def embedding(self, work_id: WorkID) -> np.ndarray:
        """Generate a vector embedding for the work.

        This is undefined for second stage models. However, this is required for first stage models
        which are required to generate work embeddings so we can retrieve candidates using some
        approximate nearest neighbors library.
        """
        raise NotImplementedError()

    def rank(self, work_id: WorkID, candidates: List[WorkID]) -> List[Tuple[WorkID, float]]:
        """Select most relevant candidates.

        The base class simply returns the candidates in some random order.
        """
        candidates = list(candidates)
        shuffle(candidates)
        return [(candidate, i) for i, candidate in enumerate(candidates)]