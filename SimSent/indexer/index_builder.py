import gc
import os
import os.path as p
import faiss
import numpy as np
from pathlib import Path
import sqlitedict as sqld
from typing import Tuple, Union

from SimSent.vectorizer.sentence_vectorizer import SentenceVectorizer

__all__ = ['IndexBuilder']


class IndexBuilder:
    # Return Types
    MMAP_ARRAYS = Tuple[np.array, np.array, np.array]
    BASE_INDEX = faiss.Index

    def __init__(self, project_dir: Union[str, Path],
                 sentence_vectorizer: object = None, large_encoder: bool = False):
        self.project_dir = Path(project_dir)
        self.sub_dir = None
        self.seed_name = None

        if sentence_vectorizer:
            self.sv = sentence_vectorizer
        else:
            self.sv = SentenceVectorizer(large=large_encoder)

    def tsv_to_index(self, dump_tsv: Union[str, Path],
                     compression: str = 'SQ8'):     # SQ8 for speed
        f_name = Path(dump_tsv).stem
        self.sub_dir = self.project_dir/f_name
        self.seed_name = f_name
        os.makedirs(p.abspath(self.sub_dir), exist_ok=True)

        # Vectorize to npz
        npz_name = self.sub_dir/f'{f_name}.npz'
        if not p.exists(p.abspath(npz_name)):
            self.sv.prep_npz(input_tsv=dump_tsv, output_npz=npz_name)

        # Load as mmap arrays
        ids, embs, sents = self.load_npz(npz_name)

        # 256 centroids for every 10k training examples
        n_centroids = int(divmod(len(embs), 10000)[0] * 256)

        # Prepare base index & write on-disk index
        base_index = self.train_base_index(embeddings=embs,
                                           n_centroids=n_centroids,
                                           compression=compression)
        self.make_mmap_index(base_index=base_index, embs=embs, ids=ids)

        # Get id-to-sent mapping
        self.populate_db(ids=ids, sents=sents)

    @staticmethod
    def load_npz(npz_name: Union[str, Path]) -> MMAP_ARRAYS:
        with np.load(npz_name, mmap_mode='r') as npz:
            ids = npz['ids']
            embs = npz['embs']
            sents = npz['sents']
        return ids, embs, sents

    def train_base_index(self, embeddings: np.array,
                         n_centroids: int = 512, compression: str = 'Flat'  # SQ8 or Flat
                         ) -> BASE_INDEX:
        # Note: Every x256 centroids requires 10k additional training points
        idx_type = f'IVF{n_centroids},{compression}'
        base_idx_pth = p.abspath(self.sub_dir/f'{idx_type}_base.index')

        if p.exists(base_idx_pth):
            index = faiss.read_index(base_idx_pth)
        else:
            index = faiss.index_factory(embeddings.shape[1], idx_type)
            index.train(embeddings)
            faiss.write_index(index, base_idx_pth)
        return index

    def make_mmap_index(self, base_index: BASE_INDEX,
                        ids: np.array, embs: np.array):
        # Get invlists
        index = faiss.clone_index(base_index)
        index.add_with_ids(embs, ids)
        ivf_vector = faiss.InvertedListsPtrVector()
        ivf_vector.push_back(index.invlists)
        index.own_invlists = False
        del index
        gc.collect()

        # Make MMAP ivfdata
        index_name = p.abspath(self.sub_dir/f'{self.seed_name}')
        invlists = faiss.OnDiskInvertedLists(base_index.nlist,
                                             base_index.code_size,
                                             f'{index_name}.ivfdata')
        ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())

        # Link index to ivfdata and save
        index = faiss.clone_index(base_index)
        index.ntotal = ntotal
        index.replace_invlists(invlists)
        faiss.write_index(index, f'{index_name}.index')

    def populate_db(self, ids: np.array, sents: np.array):
        db_file = p.abspath(self.sub_dir/f'{self.seed_name}.sqlite')
        id_to_sent = sqld.SqliteDict(db_file, autocommit=True)

        for i in range(ids.shape[0]):
            id_to_sent[str(ids[i])] = str(sents[i])

        id_to_sent.close()
