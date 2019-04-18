import os
import os.path as p
import json
import requests
from tqdm import tqdm
from typing import List, Union

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .base_vectorizer import BaseVectorizer


#### Query Vectorization ####
class DockerVectorizer(BaseVectorizer):
    """
    Intended for fast Query Vectorization.
    Note: Ensure docker container is running before importing class.
    """
    def __init__(self, large: bool = False, model_name: str = None):
        BaseVectorizer.__init__(self)

        if not model_name and large:
            model_name = 'USE-large-v3'
            self.large_USE = True
        elif not model_name:
            model_name = 'USE-lite-v2'
        self.url = f'http://localhost:8501/v1/models/{model_name}:predict'

    def make_vectors(self, query: Union[str, List[str]]):
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
    """
    Intended for batch Corpus Vectorization
    """
    def __init__(self, large: bool = False, path_to_model: str = None):
        BaseVectorizer.__init__(self)

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
        self.session = None
        print(f'Loading model: {self.path_to_model}')
        self.define_graph()
        print('Initializing TF Session...')
        self.start_session()

    def define_graph(self):
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.model = hub.Module(self.path_to_model)

    def start_session(self):
        self.session = tf.Session()
        with self.graph.as_default():
            self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def close_session(self):
        self.session.close()
        tf.reset_default_graph()
        self.define_graph()

    def make_vectors(self, sentences: List[str], n_minibatch: int = 512) -> List[tf.Tensor]:

        embeddings = list()
        batched_tensors = list()
        with self.graph.as_default():
            # High throughput vectorization (fast)
            if len(sentences) > n_minibatch:
                while len(sentences) >= n_minibatch:
                    batch = list(sentences[:n_minibatch])
                    sentences = list(sentences[n_minibatch:])
                    batched_tensors.append(tf.constant(batch, dtype=tf.string))

                dataset = tf.data.Dataset.from_tensor_slices(batched_tensors)
                dataset = dataset.make_one_shot_iterator()
                make_embeddings = self.model(dataset.get_next())

                while True:
                    try:
                        embeddings.append(self.session.run(make_embeddings))
                    except tf.errors.OutOfRangeError:
                        break

            # Tail end vectorization (slow)
            if len(sentences):
                basic_batch = self.model(sentences)
                embeddings.append(self.session.run(basic_batch))

        return embeddings

    @staticmethod
    def batch_generator(id_sent_tsv, batch_size: int = -1):
        ids = list()
        sents = list()
        with open(id_sent_tsv) as tsv:
            for line in tsv:
                sent_id, sent_text = line.split('\t')
                ids.append(sent_id), sents.append(sent_text)

        while len(ids):
            yield list(ids[:batch_size]), list(sents[:batch_size])
            ids = list(ids[batch_size:])
            sents = list(sent_text[batch_size:])

    def prep_npz(self, id_sent_tsv, id_sent_npz,
                 batch_size: int = 512*128, minibatch_size: int = 512):

        all_ids, all_sents, all_embs = list(), list(), list()

        batch_gen = self.batch_generator(id_sent_tsv, batch_size)
        for id_batch, sent_batch in tqdm(batch_gen):
            embs = self.make_vectors(sent_batch, minibatch_size)

            id_batch = np.array(id_batch, dtype=np.int64)
            sent_batch = np.array(sent_batch, dtype=np.str)
            emb_batch = np.vstack(embs).astype(np.float32)

            all_ids.append(id_batch)
            all_sents.append(sent_batch)
            all_embs.append(emb_batch)

        all_ids = np.vstack(all_ids)
        all_sents = np.vstack(all_sents)
        all_embs = np.vstack(all_embs)

        np.savez(id_sent_npz,
                 ids=all_ids, sents=all_sents, embs=all_embs,
                 compressed=True)


