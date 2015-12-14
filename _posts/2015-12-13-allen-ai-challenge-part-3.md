---
layout: post
title: 'Allen AI Challenge Part 3 - Word2Vec'
date: 2015-12-13 15:39
comments: true
categories:
---

## tl;dr

I ran Word2Vec on the text-only set of wikipedia, with every article on its own line. Then, I used a similar evaluation system as I did for LSA in the [last post](http://zhangbanger.github.io/2015/11/30/allen-ai-challenge-part-2.html).

I just and got **0.27750**, which was actually lower than LSA on the test set.

## Word2Vec Explained

In the last post, I mentioned that a lot of context within the content was lost. The semantic distribution of a word under LSA is only affected by which articles it appears in. We've failed to capture which words appear **near** each other, and with Word2Vec, we can remedy that.

The [original page for Word2Vec](https://code.google.com/p/word2vec/) lays out the value proposition and linking to key papers, the canonical one being [Efficient Estimation of Word Representations in
Vector Space](http://arxiv.org/pdf/1301.3781.pdf).

The title tells you quite a bit. The idea is to represent words in vector space in an efficient and useful way. Like LSA, we want to represent words in terms of their distribution. Unlike LSA, rather than representing words in terms of distribution across documents, we represent them in terms of distribution across **neighbor** words.

### Brief Digression into Softmax Regression

To obtain this estimation, we set up a [Softmax Regression problem](https://en.wikipedia.org/wiki/Multinomial_logistic_regression). Softmax Regression, or Multinomial Logistic Regression, is a generalization of Binary Logistic Regression where we try to "match" the target distribution with our output distribution. The difference between these distributions is known as [cross entropy][https://en.wikipedia.org/wiki/Cross_entropy], and we seek to minimize that difference. A visual and highly intuitive explanation of cross entropy is given [here](http://colah.github.io/posts/2015-09-Visual-Information/.

#### Cross Entropy

$$
- \sum\limits_i p(i) \log q(i)
$$

Where $p$ is the the true discrete probability distribution, $q$ is the generated distribution under the current model parameters, and $i$ is a discrete outcome.

#### Binary Logistic Regression

In the binary scenario, we'd like the outcome of random variable $Y$ is either $0$ or $1$. The true probabilities of $Y$ can be expressed as $p_{Y=1} = y$ and $p_{Y=0} = 1 - y$. If you consider flipping an unfair coin, the probability of heads is $y$ and the probability of tails is 1 minus the probability of heads, or $1 - y$. Similarly, your predicted outcome $\hat{Y}$ can be expressed as $q_{\hat{Y}=1} = \hat{y}$ and $q_{\hat{Y}=0} = 1 - \hat{y}$.

The binary cross entropy of $p$ and $q$ can be rewritten as:

$$
- y \log \hat{y} - (1 - y) \log (1 - \hat{y})
$$

where $i$ indexes 2 discrete outcomes, so the summation can be written out fully.

To generate the **prediction** for $\hat{Y}$, we use the following model:

$$
\frac{1}{1 + e^{-\theta^Tx}}
$$

where $\theta$ and $x$ are your model parameters and inputs, respectively.

This function squashes our signal to a range between 0 and 1, centered at 0.5, and has a lot of other interesting properties that you can read up on. It's also known as the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) and it represents "log odds" rather than a log probability (subtle difference that makes the math work out nicely).

#### Generalization to Softmax

Now let's generalize to $k$ classes. Because we know a probability distribution must sum to 1, we can do some algebra on the log odds described above (omitted) and arrive at:

$$
p(y = j \mid x;\theta) = \frac{e^{\theta_j^Tx}}{\sum\limits_{i=1}^k e^{\theta_i^Tx}}
$$

where $j$ is the current class label and the denominator represents normalization by summing up all possible numerators (all possible class labels), indexed by $i$.

Let's bring it all together! We find model parameters $\theta$ that minimize the cross-entropy loss function across all $k$ classes:

$$
\theta = argmin \Bigg(-\frac{1}{k}\bigg[\sum\limits_{i=1}^k y \log p(y = j \mid x;\theta) + (1 - y) \log (1 - p(y = j \mid x;\theta))\bigg]\Bigg)
$$

While there is no closed-form solution, a wealth of literature exists on computing the gradient and optimizing through [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) or [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS).

You'll see in a moment why I eschew particulars of this step and focus on other optimizations.

### The Word2Vec Softmax

Note: I'm terrible at making diagrams, so I'll be borrowing from others with credit given.

If a word can be represented by the distribution of its neighbors, and we can use softmax regression to find a way to minimize differences between distributions, then all that remains is to find a meaningful set of inputs and outputs that we can optimize over.

#### Neighbor Words a.k.a Skipgrams

Word2Vec captures context through [skipgrams](https://en.wikipedia.org/wiki/N-gram#Skip-gram), built from sequences of text in a document. A simple example is `"the fat cat sat in the hat"`. A $k$-skip-$n$ gram is where $n$ is the length of the subsequence and $k$ is maximum distance between words. If you fix $n = 2$ and set $k$ to half of the size of the context $C$ you care about, where, context is symmetric about the center word, you can generate pairs of words where the first item is the center word and the second item is the context word that is at most $k$ elements away.

Here, the input word is `"sat"`, and the pairs are `("sat", "the"), ("sat", "fat"), ("sat", cat"), ("sat", "in"), ("sat", "the"), ("sat", "hat")`. $k$ is 4, so context size $C$ is 8. These can form the input-output pairs for softmax regression, where the input vector $x_i$ is a one-hot vector representing the $i^{th}$ word of the vocabulary $V$, and the output vector $y_{k, j}$ is the one-hot vector for the $k^{th}$ context word in context $C$ and $j^{th}$ word of the same vocabulary $V$.

#### Hidden Layer Projection

Having input-output pairs is insufficient, because you aren't doing anything meaningful in terms of representing the distribution. What you need is a 'hidden' layer (in neural network parlance) in between to produce a projection.

A great diagram courtesy of this [great tutorial](http://alexminnaar.com/word2vec-tutorial-part-i-the-skip-gram-model.html):

![](http://alexminnaar.com/images/skip-gram.png)

The $V x N$ matrix $W$ maps the the one-hot representation to the $N$ dimensional representation and then back. Because these weights are optimized based on context words as outputs, this representation "embeds" the distribution over context words in the representation of the center word.

The distributions are not explicitly learned. Instead, at every iteration, we adjust the weight matrices $W$ and $W'$ in order to more closely align the distributions by way of the hidden layer. In other words, the hidden layer acts as a proxy for `(center_word, context_word)` pairs, where more likely pairs have a lower cross entropy.

#### Noise-Contrastive Estimation

If you look at the softmax equation, you'll notice that we need to calculate the numerator for every word at every iteration, since the $\sum$ in the denominator requires it. That seems a bit burdensome. Rather than a softmax with output over all words in the vocabulary $||V||$, we instead sample $k$ **noise** words from $V$. These words are presumed to not be context words, and therefore can be used as negative examples. We end up with a much smaller softmax.

This is called Noise-Contrastive Estimation (NCE), and it is a specific application of the general practice of negative sampling, which is used in other areas of statistical learning to adjust for situations like class imbalance. The [Tensorflow Article](https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html#scaling-up-with-noise-contrastive-training) dives into more detail, along with diagrams that contrast (pun not intended) the traditional approach with NCE.

![](https://www.tensorflow.org/versions/master/images/softmax-nplm.png)

VS

![](https://www.tensorflow.org/versions/master/images/nce-nplm.png)

**In effect, every iteration of NCE pushes the context word distribution closer and noise words farther. This helps the model better discriminate between context words and noise words.**

### Why Word2Vec Works

I stated several times that Word2Vec creates a latent representation of the distribution of context words. Let's ask an interesting question:

#### What is the joint distribution of 2 words?

The joint distribution of $a$ and $b$ is $p(a) \times p(b)$. Since we're dealing with log probabilities, a really cool thing happens:

$$
\log(p(a) \times p(b)) = \log p(a) + \log p(b)
$$

Remember how the latent representation was just a linear map? That means that if we want the joint distribution of 2 terms $a$ and $b$, we can just add the latent representations:

$$
joint(a,b) = W^Tx_a + W^Tx_b = v_a + v_b
$$

Where $W_{V \times N}$ is the weight matrix and $v_i$ is the word vector for word $i$ in the vocabulary.

In log space, we can perform vector addition and subtraction in order to manipulate probability distributions that we learned for word contexts. This intuitively makes sense, because the context for multiple terms should be the joint distribution between them.

A common example is: $v_{king} - v_{man} + v_{woman} \approx v_{queen}$

If you train or load up a Word2Vec model a sufficiently broad dataset, you can marvel at how well this works.

## Applying Word2Vec to Wikipedia

The tldr; is that I chose to use gensim and run Word2Vec within its framework. While gensim does not have GPU support,

(1) it works with Wikipedia in a way that makes sense to me,

(2) it has a lot of nice tools for evaluation and debugging, and

(3) I wasn't able to find a GPU implementation that didn't require me to do a lot of rewiring


### What is context?
The point made earlier about words appearing closely **within** documents is a fine one. The original approach to Word2Vec uses a dataset called `text8`, which is a text-only version of Wikipedia, trimmed to the first $10^8$ bytes. This text has no newlines or delineation between articles, sentences, or paragraphs. It's just 1 giant string.

I feel like using that dataset adds unnecessary noise, because you'll have positive examples where the context word is actually not a real context word. On the flipside, delineating by sentences or paragraphs seems a bit extreme, because artifical boundaries can lead to words that would be good context words not being included in the window.

I settled somewhere in-between, and it happens to fall in nicely with gensim's `LineSentence` approach. After removing XML and wiki tags, every article is a separate string of words. Words within a given article have the potential to be context words, irrespective of paragraph, sentence, and section boundaries.

#### Sidenote

I invested some time in the Tensorflow implementation, but a lot of Tensorflow `Ops` were not yet ported to GPU. Tensorflow implementation assumes the `text8` approach, using a single string as input, with no notion of article boundaries. There's some discussion on GitHub for contribution to Tensorflow to address the first issue, and I will log something to address the second.

### Extract

Using the Wikipedia latest set downloaded and compressed in bz2, you can read in the corpus and spit out the parsed lines. After reviewing numerous Wikipedia parsing scripts, including the perl script used to create `text8`, I found the gensim wikipedia parsing to be the best. It's included as part of the `WikiCorpus` constructor.

```python
wiki_corpus = WikiCorpus(input_articles, lemmatize=False)
wiki_lines = wiki_corpus.get_texts()

# Write wiki_lines out for future use
lines_output = open(output_lines, 'w')
for text in wiki_lines:
    lines_output.write(" ".join(text) + "\n")
lines_output.close()
```

### Train

We stream the file from disk using the convenient `LineSentence` class.

Sidenote: Related to the sidenote above, I tried hacking around the C++ in Tensorflow to stream off disk rather than reading the entire file into a string, but the abstraction was too leaky for a C++ rookie like me to handle elegantly.

```python
model = Word2Vec(
    sentences=LineSentence(wiki_lines),
    size=400,
    negative=25,
    hs=0
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count()
)
```

#### Explanation of hyperparameters

* I chose an embedding `size` of 400 because:

	(1) it was the largest recommended size for Word2Vec and recommended for the full Wiki dataset by the literature; and

	(2) it is the same dimensionality I ran for LSA, so it serves as good comparison for encoding efficiency.

* A `window` size of 5 is pretty standard, and it implies a context of 10.

* The default settings for gensim Word2vec don't use negative sampling, which I think would make this way too slow. I found a `negative=5` sampling size to be [supported by literature](http://arxiv.org/pdf/1310.4546.pdf) for large datasets. I also set `hs=0` to turn off Heirarchical Softmax just in case, even though the 2 methods should never be mixed.

* I left the `min_count` vocabulary filter at 5, meaning that terms appearing less than 5 times are stripped from the vocabulary and replaced with a special `"UNK"` token. 99% of the corpus is retained:

```
2015-12-14 04:09:35,030 : INFO : collected 8322355 word types from a corpus of 2125689029 raw words and 3938764 sentences
2015-12-14 04:09:48,999 : INFO : min_count=5 retains 2037044 unique words (drops 6285311)
2015-12-14 04:09:48,999 : INFO : min_count leaves 2116089449 word corpus (99% of original 2125689029)
```

* I applied subsampling of frequent words so the model spends less time on pairs such as `("the", "France")`. The value `sample=1e-5` was chosen based on the [same article mentioned in the last bullet](http://arxiv.org/pdf/1310.4546.pdf). Downsampling futher cuts down on the corpus, focusing on elements that don't add value to the model.

```
2015-12-14 05:21:36,320 : INFO : sample=1e-05 downsamples 4157 most-common words
2015-12-14 05:21:36,320 : INFO : downsampling leaves estimated 853706722 word corpus (40.3% of prior 2116089449)
```

With those settings, I'm able to get a sustained 150k words/sec on 8 threads. Not bad. See [comparable Tensorflow performance on CPU](https://github.com/tobigithub/tensorflow-deep-learning/wiki/word2vec-example).

Finally, there's a fun little test you can do with some more analogies like the "king-queen" one I gave above.

```python
demo_questions = sys.argv[3] # question-words.txt analogy example
...
# Evaluate using analogy file:
# https://word2vec.googlecode.com/svn/trunk/questions-words.txt
model.accuracy(open(demo_questions))
```

It seems to have done pretty well:

```
2015-12-14 06:55:59,456 : INFO : capital-common-countries: 91.9% (465/506)
2015-12-14 06:56:37,776 : INFO : capital-world: 93.2% (2198/2359)
...
2015-12-14 06:59:33,232 : INFO : total: 70.7% (9459/13384)
```

## Results

The code looks similar to the LSA evaluation script, with different libraries plugged in. 
The operative sections are loading the model, generating the model representation, and comparing the answers:

```python
from utils import answer2idx, idx2answer
...
model = gensim.models.Word2Vec.load(input_model)
...
answer_dict = {idx2answerchar(idx): answer for idx, answer in answers}
similarities = {
  answer_char:
    model.n_similarity(
      question_words,
      answer.translate(None, string.punctuation).split()
    )
  for answer_char, answer
  in answer_dict
got_right_answer = correct_answer == chosen_answer
correct += got_right_answer
total += 1
```

Here are results so far:

| Method | Accuracy |
|-------------|-----:|
| BoW     | 0.2432 |
| LSA       | 0.2728|
| Word2Vec    | **0.2824**|


### Intuition & Final Thoughts

Word2Vec doesn't dramatically outperform LSA. Some more hyperparameter tuning may help, but it isn't immediately obvious that this is a better distributional hypothesis than LSA.

I'm still convinced that Wikipedia is a pretty good source, so the next level of improvement will come from taking steps beyond semantic inference. Instead of semantic similarity between question and answer, we'll look at Sequence-to-Sequence learning via Recurrent Neural Networks (RNNs). RNNs are trained on pairs of input-output sequences of variable length.

The intuition behind RNNs is that dependences in the structure of language can be modeled when the inputs and outputs are arranged in a sequence and the model is forced to learn weights that maximize the generation of the output sequence given the input sequence. Stay tuned.

The code for this blog post can be found [here](https://github.com/ZhangBanger/allen-ai-challenge/tree/v10).

Final sidenote: Some cool examples are illustrated by [character level LSTMs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), although some side experimentation with the entire Wikipedia dataset at the character level proved difficult to tune.
