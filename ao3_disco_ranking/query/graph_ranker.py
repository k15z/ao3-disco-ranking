import typing
from collections import Counter
from typing import Iterable, Optional, Sequence, Tuple

from ao3_disco_ranking.types import WorkID


class GraphRanker:
    def __init__(self, collections):
        self.collections = collections

    def rank(
        self,
        work_id: WorkID,
        allowlist: Optional[Iterable[WorkID]] = None,
        blocklist: Iterable[WorkID] = [],
        num_results: int = 50,
    ) -> Sequence[Tuple[WorkID, float]]:
        """Rank works based on bookmark score.

        Args:
            work_id: The work ID that is being queried.
            allowlist: A list of works to consider. If not set, all works are considered.
            blocklist: A list of works to remove.
            num_results: The number of candidates to return.

        Returns:
            A list of tuples contain the work_id and relevance score.
        """
        work_to_score: typing.Counter[str] = Counter()
        for collection in self.collections:
            if work_id in collection:
                for other_id in collection:
                    if other_id == work_id or other_id in blocklist:
                        continue
                    if not allowlist or (other_id in allowlist):
                        work_to_score[other_id] += 1
        return work_to_score.most_common(num_results)  # type: ignore
