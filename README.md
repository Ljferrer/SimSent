# SimSent
SimSent combines Google's [Universal Sentence Encoder](https://tfhub.dev/google/collections/universal-sentence-encoder/1), a Transformer NN that encodes a sentence into a high-dimensional dense vector of fixed size (the model is trained on english, but it will accept any text as input), and Facebook AI Research's Similarity Search tool, [Faiss](https://github.com/facebookresearch/faiss), "a library for efficient similarity search and clustering of dense vectors." 

The result is a (quick and dirty) search engine that uses nearest neighbor approximation to find sentences in your corpora that are representationally adjacent to any input query. 

## How it works
SimSent can be stood up with minimal effort and is intended to be useful for exploring a corpora of english text, split into sentences (phrases, titles, posts, or even paragraphs -- whatever makes sense for your corpora) that each have a unique integer ID. 

More specifically, SimSent takes an `input.tsv` where each row follows the format: 

```python
int(ID)   "Sentence text..."
```

Using this, SimSent will encode your corpora into an `.npz` containing the ID, text, and dense vector representations ([see this docstring](https://github.com/Ljferrer/SimSent/blob/4a916b09088c753e1fc950f40a016c38e7bf1217/SimSent/vectorizer/sentence_vectorizer.py#L152-L170) for more details about how the `input.tsv` is ingested). 

From there, SimSent creates a searchable faiss index that you can build, rebuild, and tune to your needs.

## Performance 
Although I have not included any formal analysis that compares SimSent to other out-of-the-box text search solutions, it anecdotally has performed much better than TF-IDF.

## Virtual Environment
#### Initialize:
```bash
conda env create .
source activate SimSent
ipython kernel install --user --name=SimSent
```
