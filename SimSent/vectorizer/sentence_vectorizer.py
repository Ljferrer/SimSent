import gc
import os
import os.path as p
import json
import requests
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from .base_vectorizer import BaseVectorizer

tf.logging.set_verbosity(tf.logging.ERROR)

__all__ = ['DockerVectorizer', 'SentenceVectorizer']


#### Query Vectorization ####
class DockerVectorizer(BaseVectorizer):
    """ Intended for online query vectorization.
    Note: Ensure docker container is running before importing class.
    """
    DOCKER_EMB = List[List[np.float32]]

    def __init__(self, large: bool = False, model_name: str = None):
        super().__init__()

        if not model_name and large:
            model_name = 'USE-large-v3'
            self.large_USE = True
        elif not model_name:
            model_name = 'USE-lite-v2'
        self.url = f'http://localhost:8501/v1/models/{model_name}:predict'

    def make_vectors(self, query: Union[str, List[str]]) -> DOCKER_EMB:
        """ Takes one query """
        if not isinstance(query, list):
            query = [str(query)]
        elif len(query) > 1:
            query = query[:1]

        payload = {"inputs": {"text": query}}
        payload = json.dumps(payload)

        response = requests.post(self.url, data=payload)
        response.raise_for_status()

        return response.json()['outputs']


#### Corpus Vectorization ####
class SentenceVectorizer(BaseVectorizer):
    """ Intended for batch vectorization of a large text corpus """
    ID_BATCH = List[int]
    SENT_BATCH = List[str]
    GEN_BATCH = Tuple[ID_BATCH, SENT_BATCH]
    EMB_BATCH = List[tf.Tensor]

    def __init__(self, large: bool = False, path_to_model: Path = None):
        super().__init__()

        model_parent_dir = p.abspath(p.join(p.dirname(__file__), 'model/'))
        if large:
            model_dir = '96e8f1d3d4d90ce86b2db128249eb8143a91db73/'
            model_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
            self.large_USE = True
        else:
            model_dir = '1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47/'
            model_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'
        model_path = p.join(model_parent_dir, model_dir)

        if path_to_model:
            self.path_to_model = p.abspath(path_to_model)
        elif p.isdir(model_path):
            self.path_to_model = model_path
        else:
            self.path_to_model = model_url
            os.makedirs(model_parent_dir, exist_ok=True)
            os.environ['TFHUB_CACHE_DIR'] = model_parent_dir

        self.graph = None
        self.model = None
        self.sess = None
        print(f'Loading model: {self.path_to_model}')
        self.define_graph()
        print('Initializing TF Session...')
        self.start_session()

    def define_graph(self):
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.model = hub.Module(self.path_to_model)

    def start_session(self):
        self.sess = tf.Session()
        with self.graph.as_default():
            self.sess.run([tf.global_variables_initializer(),
                           tf.tables_initializer()])

    def close_session(self):
        self.sess.close()
        tf.reset_default_graph()
        self.define_graph()

    @staticmethod
    def batch_generator(id_sent_tsv: Path,
                        batch_size: int = 128 * 128) -> GEN_BATCH:
        """ Generator for tf.data.Dataset object """
        ids = list()
        sents = list()
        with open(p.abspath(id_sent_tsv)) as tsv:
            for line in tsv:
                sent_id, sent_text = str(line).replace('\n', '').split('\t')
                ids.append(int(sent_id)), sents.append(str(sent_text))

        while len(sents):
            yield (list(ids[:batch_size]), list(sents[:batch_size]))
            ids, sents = list(ids[batch_size:]), list(sents[batch_size:])
            gc.collect()

    def make_vectors(self, sents: List[str],
                     minibatch_size: int = 128) -> EMB_BATCH:
        """ High throughput, GPU-friendly vectorization """
        embeddings = list()
        batched_tensors = list()
        with self.graph.as_default():

            # High throughput vectorization (fast)
            if len(sents) > minibatch_size:
                while len(sents) >= minibatch_size:
                    batch = list(sents[:minibatch_size])
                    sents = list(sents[minibatch_size:])
                    batched_tensors.append(tf.constant(batch, dtype=tf.string))

                dataset = tf.data.Dataset.from_tensor_slices(batched_tensors)
                dataset = dataset.make_one_shot_iterator()
                make_embeddings = self.model(dataset.get_next())
                while True:
                    try:
                        embeddings.append(self.sess.run(make_embeddings))
                    except tf.errors.OutOfRangeError:
                        break

            # Tail end vectorization (slow)
            if len(sents):
                basic_batch = self.model(sents)
                embeddings.append(self.sess.run(basic_batch))

        return embeddings

    def prep_npz(self, input_tsv: Path, output_npz: Path,
                 batch_size: int = 512 * 128, minibatch_size: int = 128):
        """
        Vectorizes sentences and saves id into a numpy disk array.
        Avoids redundant computation if index must be rebuilt.

        :param input_tsv: Path to input file.tsv
                * Format:   int(ID)     "sentence"

        :param output_npz: Path to output file.npz
                * Items:    ['ids']     int         (n, )
                            ['sents']   str         (n, )
                            ['embs']    float32     (n, 512)

        :param batch_size: N sent yielded by batch_generator
        :param minibatch_size: batch_size % minibatch_size should be 0

        Writes file to path: output_npz
        """
        all_ids, all_sents, all_embs = list(), list(), list()

        batch_gen = self.batch_generator(input_tsv, batch_size=batch_size)
        for id_batch, sent_batch in batch_gen:
            embs = self.make_vectors(sent_batch, minibatch_size=minibatch_size)

            id_batch = np.array(id_batch, dtype=np.int64)
            emb_batch = np.vstack(embs).astype(np.float32)
            sent_batch = np.array(sent_batch, dtype=np.str)

            all_ids.append(id_batch)
            all_embs.append(emb_batch)
            all_sents.append(sent_batch)

        all_ids = np.concatenate(all_ids, axis=None)
        all_embs = np.concatenate(all_embs, axis=0)
        all_sents = np.concatenate(all_sents, axis=None)

        np.savez(output_npz,
                 ids=all_ids, embs=all_embs, sents=all_sents,
                 compressed=True)
