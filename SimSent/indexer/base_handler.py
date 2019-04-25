import os.path as p
import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss

from SimSent.indexer.faiss_cache import faiss_cache

__all__ = ['BaseIndexHandler']


class BaseIndexHandler(object):
    DiffScores = List[np.float32]
    VectorIDs = List[np.int64]
    FaissSearch = Tuple[DiffScores, VectorIDs]

    def __init__(self):
        self.index = None
        self.dynamic = False
        self.io_flag = faiss.IO_FLAG_ONDISK_SAME_DIR

    @faiss_cache(128)
    def search(self, query_vector: np.array, k: int) -> FaissSearch:
        return self.index.search(query_vector, k)

    def get_index_paths(self, idx_dir_pth: Path,
                        nested: bool = False) -> List[Path]:

        get = '*/*.index' if nested else '*.index'
        index_paths = glob.glob(p.abspath(idx_dir_pth/get))
        index_paths = [Path(pth) for pth in index_paths if  # Skip empty indexes
                       faiss.read_index(pth, self.io_flag).ntotal > 0]

        return sorted(index_paths)

    @staticmethod
    def joint_sort(scores: DiffScores, ids: VectorIDs) -> FaissSearch:
        """
        Sorts scores in ascending order while maintaining score::id mapping.
        Checks if input is already sorted.
        :param scores: Faiss query/hit vector L2 distances
        :param ids: Corresponding faiss vector ids
        :return: Scores sorted in ascending order with corresponding ids
        """
        # Check if sorted
        if all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)):
            return scores, ids

        # Joint sort
        sorted_difs, sorted_ids = (list(sorted_dif_ids) for sorted_dif_ids
                                   in zip(*sorted(zip(scores, ids))))

        return sorted_difs, sorted_ids
