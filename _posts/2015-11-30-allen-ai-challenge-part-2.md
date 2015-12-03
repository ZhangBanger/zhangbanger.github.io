---
layout: post
title: 'Allen AI Challenge Part 2'
date: 2015-11-30 23:24
comments: true
categories:
---

## tl;dr

I just [submitted on Kaggle](https://www.kaggle.com/c/the-allen-ai-science-challenge/leaderboard#team-237126) and got **0.30625**. I haven't had a chance to test out the variance of this approach, but I'm betting it's pretty huge.

## Evaluation Script

To set up the evaluation script, we need to load the model that we trained in Part 1:

```python
# Load model
# Note - model contains dictionary that intentionally omits stopwords
model = gensim.models.LsiModel.load(input_model, mmap='r')

# Load 'training' data
training_data = open(input_training)
```

The task is to predict the correct answer from 4 choices given a question. Since this isn't a traditional classifcation problem with a threshold, we don't really care about traditional measures like AUC or F1, and there is no "curve" to measure. Instead, we measure accuracy by $\frac{numCorrect}{numTotal}$.

This is extremely simple:

```python
correct = 0
total = 0

for line in training_data:
  ...
  correct += correct_answer == chosen_answer
  total += 1
```

## Bag of Words Similarity

As an absolute baseline using the worst possible method I can think of, we should be able to get results that approximate random guessing (~25% accuracy). This version on github basically ignores the LSA projection, calculates cosine distance based on the original Bag of Words representation with some TF-IDF weighting applied.

### The Bag of Words (BoW) representation
A document `"dog run dog dog"` in the BoW representation would look like `[(1342, 3), (65, 1)]`, denoting that the "dog," the 1342^nd word in the dictionary, occurs 3 times, and "run," the 65^th word in the dictionary, occurs once. This is a sparse vector, and could also be a Python `dict` or `Map` in any other language.

After extracting some fields, we produce a bag of words representation of the question and 4 answers:

```python
doc_vectors = [model.id2word.doc2bow(element.split()) for element in elements]
question = doc_vectors.pop(0)
```

And choose the answer with the highest cosine similarity to the question:

```python
similarities = [(gensim.matutils.cossim(question, answer), chr(idx + 65)) for idx, answer in enumerate(doc_vectors)]
chosen_answer = max(similarities)[1]
```

I used a little character-ordinal trick to go between alphabetic characters and numeric indexes for the answer. As an aside, I do a lot of mutable crap in here that's generally very bad practice, but it keeps things terse and legible.

Here are the baseline results:

| Measure | Qty |
|-------------|-----:|
| Correct     | 608 |
| Total       | 2500|
| Accuracy    | **0.2432**|



The [code](https://github.com/ZhangBanger/allen-ai-challenge/tree/v2) tagged with this version on GitHub.

## LSA Similarity

Switching to LSA-projected similarity is super easy:

```python
similarities = [(gensim.matutils.cossim(model[question], model[answer]), chr(idx + 65)) for idx, answer in enumerate(doc_vectors)]
```

The only thing changes are that `question` becomes `model[question]` and `answer` becomes `model[answer]`. This convenient `model[x]` syntax in LSA allows you to pass in a sparse Bow representation and get back a dense LSA representation. I'll spare you the printout, but given that we choose $k = 400$ back in Part 1, you can imagine that we have a 400-component dense vector.

Here are the results:

| Measure | Qty |
|-------------|-----:|
| Correct     | 682 |
| Total       | 2500|
| Accuracy    | **0.2728**|


The [code](https://github.com/ZhangBanger/allen-ai-challenge/tree/v3) tagged with this version on GitHub.

## Thoughts & Intuition
You can see that the ultra-naive approach got 24% right, which is within \\(\epsilon\\) of 25%, the expected percentage we'd get through random guessing. Why is it that the LSA representation was slightly better, but not by a ton? The LSA representation rearranges the space such that words that co-occur in articles are closer in that latent space. Another way of saying this is that LSA incorporates the "distributional semantics" of words, but only very roughly.

To think about the equivalent of taking an 8^th grade science test using this strategy as a human, you could characterize the naive BoW and LSA approaches as follows:

**Naive BoW:** pick the answer with the words that overlap the most with your set. You are screwed if there's no overlap - in vector space, every word is orthogonal, so you would have 0 cosine similarity

**LSA:** pick the answer with the words that appear in the most similar documents as the words in the question.

The LSA approach is actually not an unreasonable one, but we know there's a lot of structural information missing. There's no concept of word closeness within a document and we've entirely thrown away the idea that words in a sequence follow some pattern and imply some meaning.

### Closer Look at Errors

Due to the contest rules, I can't post any specific examples on here. However, I can make a few anecdotal observations. I randomly sampled 1% of the errors and printed out the
(1) question;
(2) right answer in English;
(3) cosine similarity between the question and the right answer;
(4) my answer in English; and
(5) cosine similarity between the question and my answer.

```python
print("Question: %s" % elements.pop(0))
print("Correct Answer: %s" % elements[ord(correct_answer) - 65])
print("Correct Answer similarity: %.4f" % similarities[ord(correct_answer) - 65][0])
print("Chosen Answer: %s" % elements[ord(chosen_answer) - 65])
print("Correct Answer similarity: %.4f\n" % similarities[ord(chosen_answer) - 65][0])
```

The [code](https://github.com/ZhangBanger/allen-ai-challenge/tree/v4) tagged with this version on GitHub.

Here are some broad, qualitative cases I spotted:

**Case 1:** similarity in chosen vs correct was extremely close (0.0748 vs 0.0792) and even the answers were plausibly in the same documents. There's a question about examples of dog behavior, and the 2 examples sound very similar to me. It's only through reasoning through a supplemental concept that I arrived at the right answer.

**Case 2:** the correct answer had low similarity, but it tended to contain terms more ambiguous than the chosen answer. There's a question about structures in the heart, and out of the 1-word answers, this model chose the one most specific to hearts ("aorta") rather than a word that was used in other contexts ("valve").

**Case 3:** the most degenerate; similarity was 0 in many answer choices due to stopword being filtered from the model's dictionary. Some words are also missing from the dictionary, and numeric tokens are filtered out as well. This is where some traditional NLP preprocessing backfires. Filtering out stopwords and either "anonymizing" or ignoring numbers leads to ocassional loss of relevant information. The dictionary-size parameter might also need some tuning, with the caveat that more tolerant vocabulary rules might introduce more noise.

## Next Steps

To improve upon distributional semantics (check out [Ilya Sutskever](https://archive.org/details/Redwood_Center_2014_02_12_Ilya_Sutskever) explaining that term in a great talk), we can learn a better representation using `word2vec`. For a preview; `word2vec` also gives you a contextual representation of a word, but it focuses on neighboring words as context rather than the document space. A word can also be its own neighbor! `word2vec` doesn't explicitly cases 2 and 3, but it might be useful in applying a more fine-grained sense of context. There's also a method for minor preprocessing for word2vec where digits are converted to space-delimited, spelled-out numbers; definitely worth a try. Stay tuned.
