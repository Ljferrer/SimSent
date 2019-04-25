import os.path as p
from time import sleep
from pathlib import Path
from typing import Dict, List, Tuple, Union
from multiprocessing import Pipe, Process, Queue

import faiss
import numpy as np

from SimSent.indexer.faiss_cache import faiss_cache
from SimSent.indexer.base_handler import BaseIndexHandler

__all__ = ['RangeShards']


#### Parallelized Nearest Neighbor Search ####
class RangeShards(BaseIndexHandler):
    DiffScores = List[np.float32]
    VectorIDs = List[np.int64]
    FaissSearch = Tuple[DiffScores, VectorIDs]
    FaissMultiSearch = Dict[str, FaissSearch]

    def __init__(self, shard_dir: Union[str, Path],
                 nprobe: int = 4, get_nested: bool = False):
        """
        For deploying multiple, pre-made IVF indexes as shards.
            (intended for on-disk indexes that do not fit in memory)

        Note: The index shards must be true partitions with no overlapping ids

        :param shard_dir: Dir containing faiss index shards
        :param nprobe: Number of clusters to visit during search
                       (speed accuracy trade-off)
        :param get_nested: Load indexes in sub directories of shard_dir
        """
        super().__init__()
        self.paths_to_shards = self.get_index_paths(Path(shard_dir),
                                                    nested=get_nested)
        self.nprobe = nprobe
        self.dynamic = True
        self.lock = False

        self.results = Queue()
        self.shards = dict()
        self.n_shards = 0
        for shard_path in self.paths_to_shards:
            self.load_shard(shard_path)
        for shard_name, (handler_pipe, shard) in self.shards.items():
            shard.start()
            self.n_shards += 1

    def load_shard(self, shard_path: Path):
        shard_name = Path(shard_path).stem
        shard_pipe, handler_pipe = Pipe(duplex=False)
        shard = Shard(shard_name, shard_path,
                      input_pipe=shard_pipe,
                      output_queue=self.results,
                      nprobe=self.nprobe, daemon=False)
        self.shards[shard_name] = (handler_pipe, shard)

    @faiss_cache(128)
    def search(self, query_vector: np.array, keys: list,
               radius: float = 1.0) -> FaissMultiSearch:

        if query_vector.shape[0] > 1:
            query_vector = np.reshape(query_vector, (1, 512))

        # Lock search while loading index or actively searching
        while self.lock:
            sleep(1)

        # Lock out other searches
        self.lock = True

        # Start parallel range search
        n_results = 0
        for shard_name, (hpipe, shard) in self.shards.items():
            if shard_name in keys:
                hpipe.send((query_vector, radius))
                shard.run()
                n_results += 1

        # Aggregate results
        results = dict()
        while n_results > 0:
            name, difs, ids = self.results.get()
            results[name] = self.joint_sort(difs, ids)
            n_results -= 1

        self.lock = False
        return results


class Shard(Process):
    def __init__(self, shard_name: str, shard_path: Path,
                 input_pipe: Pipe, output_queue: Queue,
                 nprobe: int = 4, daemon: bool = False):
        """ RangeShards search worker """
        super().__init__(name=shard_name, daemon=daemon)
        self.input = input_pipe
        self.index = faiss.read_index(p.abspath(shard_path),
                                      faiss.IO_FLAG_ONDISK_SAME_DIR)
        self.index.nprobe = nprobe
        self.output = output_queue

    def run(self):
        @faiss_cache(64)
        def neighborhood(index, query, radius):
            _, _difs, _ids = index.range_search(query, radius)
            return _difs, _ids

        if self.input.poll():
            (query_vector, radius_limit) = self.input.recv()
            difs, ids = neighborhood(self.index, query_vector, radius_limit)
            self.output.put((self.name, difs, ids), block=False)
