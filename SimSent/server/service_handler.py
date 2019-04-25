import os.path as p
import glob
from time import time
from pathlib import Path
from sqlitedict import SqliteDict
from typing import Dict, List, Tuple, Union

import numpy as np

from SimSent.indexer.faiss_cache import faiss_cache

__all__ = ['QueryHandler']


class QueryHandler:
    QueryReturn = np.array
    DiffScores = List[np.float32]
    VectorIDs = List[np.int64]
    FaissSearch = Tuple[DiffScores, VectorIDs]
    FormattedSearch = List[Tuple[np.int64, np.float32, str]]
    FormattedMultiSearch = Dict[str, FormattedSearch]

    def __init__(self, query_vectorizer: object, index_handler: object,
                 project_dir: Path, nested: bool = False):
        super().__init__()
        self.vectorizer = query_vectorizer
        self.indexer = index_handler

        # Get id-to-sent maps
        get = '*/*.sqlite' if nested else '*.sqlite'
        db_files = glob.glob(p.abspath(project_dir / get))
        self.sent_dbs = dict()
        for f in db_files:
            self.sent_dbs[Path(f).stem] = SqliteDict(f)

    @faiss_cache(32)
    def query_corpus(self, query_str: str, keys: List[str],
                     k: int = 5, radius: float = 1.0, verbose: bool = True
                     ) -> FormattedMultiSearch:
        """
        Vectorize query -> Search faiss index handler -> Format doc payload.
        Expects to receive only one query per call.
        """
        # Vectorize
        t_v = time()
        query_vector = self.vectorize(query_str)

        # Search
        t_s = time()
        results = self.indexer.search(query_vector, keys, radius=radius)

        t_p = time()
        top_hits = list()
        similar_docs = dict()
        for source, result_set in results.items():
            sorted_set = self.format_results(source, result_set, k)
            top_hits.extend(sorted_set)
            similar_docs[source] = sorted_set
        similar_docs['top_hits'] = sorted(top_hits)[:k]

        t_r = time()
        if verbose:
            print(f'  Query vectorized in --- {t_s - t_v:0.4f}s')
            print(f'  Index searched in ----- {t_p - t_s:0.4f}s')
            print(f'  Payload formatted in -- {t_r - t_p:0.4f}s\n')

        return similar_docs

    def vectorize(self, query: Union[str, List[str]]) -> QueryReturn:
        """
        Use DockerVectorizer for fast Query Vectorization.
        :param query: Text to vectorize
        :return: Formatted query embedding
        """
        if not isinstance(query, list):
            query = [query]
        if len(query) > 1:
            query = query[:1]

        query_vector = self.vectorizer.make_vectors(query)

        if isinstance(query_vector[0], list):
            query_vector = np.array(query_vector, dtype=np.float32)
        return query_vector

    def format_results(self, source: str, result_set: FaissSearch, k: int
                       ) -> FormattedSearch:
        scores, hit_ids = result_set
        sents = list()
        for sent_id in hit_ids:
            sents.append(self.sent_dbs[source][str(sent_id)])

        return sorted(zip(scores, hit_ids, sents))[:k]