---
layout: post
title: 'Allen AI Challenge Part 4 - Word2Vec Tuning'
date: 2015-12-14 19:48
comments: true
categories:
---

## Tuning Word2Vec

In my first run of Word2Vec, I ran a skip-gram model using the negative sampling optimization. Results were a bit disappointing. Compared to the LSA model, Word2Vec model actually performed worse on the Allen AI validation set, and didn't significantly beat it on the training set (althought to reiterate, the model never saw the training set either).

### Hierarchical Softmax

To further try out ideas from this [paper](http://arxiv.org/pdf/1310.4546.pdf), I will switch from negative sampling to Hierarchical Softmax (HS). HS is another way of avoiding the full computation of the softmax denominator. Rather than using a fixed sample size like in NCE, HS uses a binary tree for words based on [Huffman Coding](https://en.wikipedia.org/wiki/Huffman_coding).

A Huffman-coded word is represented by the path down the tree to that word, with the most frequent words at the top of the tree and the least frequent at the bottom. This minimizes the average encoding length of the word. In the context of Word2Vec softmaxes, a word is encoded as joint probability of taking the correct $left$ or $right$ path at every node in the tree to arrive at the word. The time complexity is reduced to $\log_2(n)$.

A Kaggle submission at this stage yielded **0.30250** on the validation set.

### Log Probability Scoring of Question + Answer

I did a quick attempt at using the `model.score()` feature in gensim where a log probability is computed for a sequence. I passed in the `question + answer` as the string, with a bit of preprocessing to tokenize, and I picked the answer with the highest score.

```python
model.score([preprocess_text_gensim(model, "%s %s" % (question, answer))])
```

It did not do well (see below), so I didn't bother submitting this one.

### Phrase2Vec - Bigrams

A common practice to help performance in Word2Vec is to apply a layer of phrase detection before training. Bigrams occur in proportion to the joint probability of individual words, which we can model like:

$$
score(w_i, w_j) = \frac{count(w_iw_j) - \delta}{count(w_i) \times count(w_i)}
$$

This is described in the [paper](http://arxiv.org/pdf/1310.4546.pdf) mentioned above. $\delta$ is a min count below which bigrams will never be considered.

A clever thing about this technique is that it can be applied recursively, where a bigram on an existing bigram is effectively a trigram.

Gensim packages this [nicely](https://radimrehurek.com/gensim/models/phrases.html#module-gensim.models.phrases).

```python
bigram_transformer = Phrases(wiki_lines)

model = Word2Vec(
    sentences=LineSentence(bigram_transformer[wiki_lines]),
...
```

It's an extra hour of single-core preprocessing, but worth it for the extra model performance.

```
2015-12-14 22:16:37,783 : INFO : collected 18485398 word types from a corpus of 2121750265 words (unigram + bigrams) and 3938764 sentences
2015-12-14 22:16:37,783 : INFO : merging 18485398 counts into Phrases<0 vocab, min_count=5, threshold=10.0, max_vocab_size=40000000>
2015-12-14 22:16:45,411 : INFO : merged Phrases<18485398 vocab, min_count=5, threshold=10.0, max_vocab_size=40000000>
```
Accuracy on the training set was reasonable (see below), but abysmal on the validation set: **0.22750**.

For reference, here is this version on [github](https://github.com/ZhangBanger/allen-ai-challenge/tree/word2vec-phrase)

### All Defaults + 2 Epochs

I hadn't given this a shot the first time around. I'm removing the Phrase transform as well. You'll notice in the code I haven't parameterized these hyperparameters; this is intentional, because I want to memorialize different versions in code rather than through some history of command line args.

For extra fun, I'll run through 2 epochs. An epoch is one pass through the data, and by default, gensim only does 1 epoch. In effect, more adjustment will be made to the weight matrix to further fine tune the word vectors. I don't expect this to have a huge effect, but I noticed that a lot of the missed answers were from relatively close cosine similarities, meaning that the word vectors were at least in the right ball park.

Results were on par with our best run, which used hierarchical sampling with common word subsampling. Submission to Kaggle yielded **0.23125**.

This final version is also [tagged on github](https://github.com/ZhangBanger/allen-ai-challenge/tree/word2vec-base). The subsampled version is also [tagged](ttps://github.com/ZhangBanger/allen-ai-challenge/tree/word2vec-hs-sub).

## Results

| Method | Accuracy |
|-------------|-----:|
| BoW - One-Hot    | 0.2432 |
| LSA       | 0.2728|
| Word2Vec - NCE-5    | 0.2824|
| Word2Vec - HS + Subsample | **0.3056** |
| Word2Vec - HS + Subsample + LogProb Scoring | 0.2592 |
| Word2Vec - HS + Subsample + Bigram | 0.2976 |
| Word2Vec - HS + 2 epoch | 0.3000 |


## Final Thoughts

While the distributional hypothesis is very interesting and has lots of support in literature and data, leaning on it only got us small improvements beyond what we might see due to statistical noise.

There's been some discussion in the forums about question-answer models and automated dependency parsing. That's something I'll look into after trying sequence-to-sequence modeling.
