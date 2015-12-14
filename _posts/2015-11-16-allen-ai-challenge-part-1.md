---
layout: post
title: 'Allen AI Challenge Part 1'
date: 2015-11-16 21:39
comments: true
categories:
---

## Background

Thanks in part to some amazing encouragement from friends, family, and mentors, I've decided to blog about tackling a Kaggle contest to demonstrate my data science chops to the world in a shareable, public way.

I chose the [Allen AI Challenge](https://www.kaggle.com/c/the-allen-ai-science-challenge) because it intersects nicely with things I've worked on around NLP and knowledge bases, but also crosses into new, interesting territory.

## Problem Statement

The question, posed somewhat pejoratively: Is your model smarter than an 8th grader?

The task for the machine is to identify the correct answer out of 4 candidate answers given a question. It is essentially the multiple-choice exams that many of us had back in 8th grade. The domain is limited to 8th grade science.

## Initial Impressions

I first checked out the forums to see what everyone were discussing, which might help direct my search later on. The topics ranged from "work with me plz" to "deep learning didn't work" to "where do I get data?" to "building a knowledge base & search." I naturally climbed down the rabbit hole of some of these discussions, effectively doing a literature review on Q-A (Question-Answer) systems and some recent work. Most in deep learning, most revolve around recurrent neural networks and memory networks. Some interesting paraphrases acquired through forums/reading:

* Developing complex feature engineering pipelines to preprocess these questions is hard and likely to cause you to overfit.
* The training set inadequately covers the topic of 8th grade science, and getting the right training data is hard. Furthermore, the question-answer pairs have to be in your training data in some form, but they aren't really found anywhere.

Armed with prior analysis of others, I checked out the data and confirmed the above. While the contest rules prohibit me from sharing the data, I can summarize; the questions require research and critical thinking for an adult human. The training set is merely an example / sanity test for building the model rather than a supervised training set. This is obvious after you look at it, since the choice of A, B, C, or D isn't really a class in terms of classification.

## Rationale for Approach

Being biased towards automation and ML, approaches that involve more feature and data engineering seemed less interesting. Up front, I decided against hunting down and slamming hundreds of textbooks into an ElasticSearch cluster. I think the current champion of the leaderboard took that approach.

Instead, I sought to apply a naive hypothesis to an accessible, simple dataset for initial insight. I plan to map the bag-of-words query to a latent space and my answers to the same space, and choose the bag-of-words answer most similar to the query, by some measure such as cosine similarity. I'd generate the mapping to the latent space through a publicly available data source like wikipedia, as required by the rules. While almost certain this model will perform at the level of random guessing, I'm planning to alternate iterating on the model and data source.

## First Run

Given the goal of minimizing data engineering, a preference for using existing tools, and comfort with any language/paradigm, I chose to use gensim, as it had a very complete suite of prebuilt tools , [Latent Semantic Analysis](https://en.m.wikipedia.org/wiki/Latent_semantic_analysis). LSA embodies what I descibed above, and while the model is training, I can go into more detail.

I downloaded the entire wikipedia dataset (which has much better coverage than the text8 set that only has the first 100 million characters and isn't segmented by document). Following this [guide](https://radimrehurek.com/gensim/wiki.html), I ran that through their preprocessor to generate a `Corpus`, which consists of a term index and a term-document matrix:
`python -m gensim.scripts.make_wiki`
The term-document matrix is essentially the bag-of-words count for every term in every document, with a pass of [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

To build the model, you need to:

- **load the term index**

```python
id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File('./data/wiki_en_wordids.txt.bz2'))
```

- **load term-document matrix**

```python
mm = gensim.corpora.MmCorpus('./data/wiki_en_tfidf.mm')
print(mm)
```

You can see that my wikipedia matrix has 3.9M documents and 100K terms. I'm sticking with default hyperparameters here for now, which means I've trimmed to the top 100K terms.

```bash
# 2015-11-16 20:05:43,304 : INFO : loaded corpus index from ./data/wiki_en_tfidf.mm.index
# 2015-11-16 20:05:43,304 : INFO : initializing corpus reader from ./data/wiki_en_tfidf.mm
# 2015-11-16 20:05:43,305 : INFO : accepted corpus with 3933461 documents, 100000 features, 612118814 non-zero entries
# MmCorpus(3933461 documents, 100000 features, 612118814 non-zero entries)
```

- And then train:

```python
num_topics = 400
lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=num_topics)
```

I'm running this while writing this article, and since it takes forever on my laptop, I'll explain Latent Semantic Analysis.

## Latent Semantic Analysis

Latent Semantic Analysis enables you to map a high-dimensional, bag-of-words representation of the counts of every word in your vocabulary across every document (wiki article) to a lower dimension representation. Your original representation is sparse because most words don't appear in most wikipedia articles, and your resulting representation is dense because every component, or latent factor, in your low dimension representation is a linear combination of words.

**The intuitive explanation of the hypothesis.**
We assume that words that occur in the same documents tend to be related, and if we believe the relationships are linear, we can manipulate (rotate, scale, rotate) the original matrix to capture those relationships.

More specifically, there exists some bases for representing the count of words across documents such that every component is orthogonal (uncorrelated). Through some series of imputed transformations (rotate, scale, rotate), which we will seek to invert, this orthogonal representation generated the data that we have.

The wikipedia diagram shows it best:

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Singular-Value-Decomposition.svg/220px-Singular-Value-Decomposition.svg.png)

On the top right is the real data, and on the top left is the desired representation. The arrows show the imputed transformation that we want to recreate and invert.

Furthermore, the columns of the matrix are ordered such that the first column encodes the most collinearity in the data, all the way down to the last column. Given such a representation, we can choose a vector length that encodes the pattern of variation in a smaller space. Assumptions here are pretty naive, and things like word order, context, dependencies, and so forth are ignored. Simple data, simple model.

**Term-Document Matrix**
The term-document matrix contains the sparse representation of occurences of words in wikipedia articles. It is weighted using TF-IDF, based on the assumption that words that appear more frequently in a given article but less frequently overall are more significant to that article, whereas words that appear in all articles add little information. This assumption makes sense from an information theory standpoint, but there's been some study around why it doesn't work. Nonetheless, it's still an industry-standard practice.

$TFxIDF$ is defined as the raw term frequency of a term in a given document times the log of the quotient of the total number of documents divided by the number of documents containing that term

$$log \dfrac{totalNumDocs}{numDocsWithTerm}$$

The final term-document matrix $X$ is a matrix where element $(i,j)$ is the $TFxIDF$ weighted occurrence of term $i$ in document $j$.

### Singular Value Decomposition

In order to represent this matrix in a useful latent space, we need construct linear map(s) that represent the data in a dense form. LSA uses [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition), a method which factorizes the matrix $X$into 3 matrices with certain properties:

$$
X = U\Sigma V^T
$$


* $U$ is a unitary matrix (orthogonal in real space) contains columns called left-singular vectors, which are the eigenvectors of $XX^T$
* $V$ is a unitary matrix (orthogonal in real space) contains columns called right-singular vectors, which are the eigenvectors of $X^TX$
* $\Sigma$ is a diagonal matrix where the diagonal elements are called singular values, which are the square roots of the eigenvalues of $X^TX$
* $X^TX$ and $XX^T$ are covariance matrices, and [it's provable that the eigenvalues are the variances](http://math.stackexchange.com/questions/1217862/why-the-principal-components-correspond-to-the-eigenvalues) (link talks about PCA, which uses SVD under the hood). These represent term and document covariances.

**Power Iteration**
The eigenvalues/eigenvectors are learned using an algorithm called [Power Iteration](https://en.wikipedia.org/wiki/Power_iteration). Power Iteration produces the eigenvalue $\lambda$ and eigenvector $v$ such that $Xw = \lambda w$. It actually doesn't decompose a matrix directly, but is a neat trick that avoids computing the covariance matrix $XX^T$, and is based on certain assumptions that you can read upon.
The tldr; you can converge on the largest eigenvalue by repeatedly multiplying your matrix $X$ by your vector $w_k$, normalizing and substituting the value of $w_k$ back in for the next iteration, in this recurrent relation:

$$
w_{k+1} = \dfrac{Xw_k}{||Xw_k||}
$$

**Elimination**
The result of a run is an eigenvalue that represents the variance in this component and an eigenvector $w_n$, where each $w_{(i)}$ component is a weight representing the contribution of term $i$ in the term-document matrix $X$ to the component. The actual component can be interpretted the linear combination of the terms that inherits the maximum possible variance.

To get $k^{th}$ eigenvalue/eigenvector, you can apply the first $k - 1$ transformations and replace the $X$ matrix above with the resulting $\hat{X_k}$.

$$
\hat{X_k} = X - \sum\limits_{n=1}^{k-1} Xw_{(n)}w{(n)}^T
$$

You can repeat this process from the 1st component all the way to the end.

**Dimensionality Reduction**
With our SVD in hand, we can choose a $k$ such that $k$ encodes the desired variance in the data in a smaller space. When we replace the middle matrix $\Sigma$ with $\Sigma_k$, where $\Sigma_k$ is $Sigma$ with only the $k$ largest singular values (all others zeroed out), we end up with a lower-rank matrix:
$$
X_k = U\Sigma_k V^T
$$

**"Inference"**
Using the `id2word` term index, you can get the vector index $i$ of a word, and with a linear combination of those, weighted by TF-IDF, you can calculate a query  vector $q$ for a new sentence.

To obtain a $k$-dimensional representation $\hat{\textbf{q}}$ of $q$, you would multiply the pseudoinverse of your lower rank $\Sigma_k$ with your singular vectors in $U$, and apply that transformation to $q$

$$
\hat{\textbf{q}} = \Sigma_k^{-1} U_k^T q
$$

This is essentially an inversion of the transformation that is implied by SVD. I'm calling this "prediction" even though it really isn't.

**Use in multiple choice**
Given:

* A latent representation of the query for the Question $q$
* The latent representations of the 4 different answers ${a_i} : i \in \{a, b, c, d\}$

Compute a [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) for reach question-answer pair, defined as the dot product normalized by the product of the norms:

$$
\cos(a, b) = \dfrac{q \cdot a_i}{||q||\times||a_i||}
$$

Choose the option with the highest similarity $i = argmax(\cos(q, a_i))$

**Notes**
Due to how gensim trains, you choose the $k$ parameter up front, as it is incorporated into the training. A quasi-online version of SVD is performed, incrementally building $U\Sigma V$ by adding small sets of documents and applying a few power iterations. Only a rank $k$ approximation is retained at every run. This technique allows it to avoid holding too much in memory at once, but it is painfully slow (6+ hours and counting on the entry-level 2015 MacBook Pro). There is some literate showing that only maintaining the low rank approximation is sufficient for this approach.

Here's a snippet of logs:

```bash
# 2015-11-16 21:29:21,843 : INFO : preparing a new chunk of documents
# 2015-11-16 21:29:22,448 : INFO : using 100 extra samples and 2 power iterations
# 2015-11-16 21:29:22,448 : INFO : 1st phase: constructing (100000, 500) action matrix
# 2015-11-16 21:29:25,071 : INFO : orthonormalizing (100000, 500) action matrix
# 2015-11-16 21:30:02,235 : INFO : 2nd phase: running dense svd on (500, 20000) matrix
# 2015-11-16 21:30:06,441 : INFO : computing the final decomposition
# 2015-11-16 21:30:06,441 : INFO : keeping 400 factors (discarding 6.190% of energy spectrum)
# 2015-11-16 21:30:08,088 : INFO : merging projections: (100000, 400) + (100000, 400)
# 2015-11-16 21:30:17,741 : INFO : keeping 400 factors (discarding 1.338% of energy spectrum)
# 2015-11-16 21:30:20,817 : INFO : processed documents up to #1040000
# 2015-11-16 21:30:20,820 : INFO : topic #0(120.686): 0.300*"median" + 0.266*"population" + 0.258*"households" + 0.247*"income" + 0.245*"census" + 0.200*"average" + 0.197*"females" + 0.196*"males" + 0.191*"families" + 0.180*"living"
# 2015-11-16 21:30:20,823 : INFO : topic #1(88.537): 0.137*"album" + 0.100*"band" + -0.093*"median" + 0.090*"station" + 0.087*"song" + -0.077*"households" + 0.075*"party" + 0.073*"game" + -0.070*"income" + 0.062*"church"
# 2015-11-16 21:30:20,827 : INFO : topic #2(61.042): -0.434*"album" + -0.268*"band" + -0.225*"song" + -0.182*"vocals" + -0.176*"chart" + -0.172*"guitar" + -0.129*"albums" + -0.114*"track" + -0.109*"songs" + -0.103*"you"
# 2015-11-16 21:30:20,831 : INFO : topic #3(50.692): -0.413*"station" + 0.214*"party" + -0.198*"town" + -0.180*"railway" + -0.163*"river" + -0.157*"village" + 0.133*"election" + -0.120*"municipality" + -0.117*"route" + -0.117*"road"
# 2015-11-16 21:30:20,834 : INFO : topic #4(48.626): -0.294*"party" + 0.277*"league" + 0.221*"game" + -0.189*"election" + 0.180*"cup" + 0.164*"football" + 0.149*"club" + 0.144*"games" + -0.133*"album" + 0.127*"player"
```
So far, looks sane....

## GitHub & Next
Stay tuned for another post on results.

[GitHub repo](https://github.com/ZhangBanger/allen-ai-challenge/tree/v1) with code tagged with every blog post (in this case, `v1`)


