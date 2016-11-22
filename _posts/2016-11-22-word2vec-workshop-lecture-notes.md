---
layout: post
title: 'Word2Vec Workshop Lecture Notes'
date: 2016-11-22 13:43
categories:
---

The following is a set of lecture notes for a workshop I taught this past weekend on Word2Vec:


# Introduction

We start with the motivating example:

## Word Vectors

### Synonyms

$$ \vec {happiness} \approx \vec {joy} $$

$$ \vec {dog} \approx \vec {puppy} $$

And a seemingly weird example that will make sense to you later:

$$ \vec {one} \approx \vec {two} $$

### Analogies

You can do arithmetic and get analogies!

$$ \vec {king} - \vec {man} + \vec {woman} \approx \vec {queen} $$

$$ \vec {Berlin} - \vec {Germany} + \vec {France} \approx \vec {Paris} $$

A visualization lifted from the web (http://blog.christianperone.com/wp-content/uploads/2016/01/word2vecqueen.png):

![Word2Vec Example](http://blog.christianperone.com/wp-content/uploads/2016/01/word2vecqueen.png)

## Magic? Nope!

Does this seem like magic to you? It might, but Word2Vec is learning what we call 'embeddings' for words, and embeddings are actually quite simple.
From the examples above, you might conclude that Word2Vec understands what an analogy is, as well knowing that Berlin is a city and Germany is a country. That's actually *not* the case!



## Boring Notation

**Notation is important!** The ability to formalize and express these concepts in a structured way helps you *truly* grasp the relationship between concepts you *think* you know and form connections to new ones.

$V$ is a matrix

$\vec v_c$ is the $c^{th}$ column vector taken from matrix $V$

$y_i$ is the $i^{th}$ scalar element of the vector $y$

$x$ is a vector

|W| is the length of the vocabulary

$v^Tu$ is the dot product of $v$ and $u$

# Embeddings

Often used interchangeably, the terms *word vectors* and *word embeddings* refer to a vector representation of a word with some (learned) semantic meaning.

## Questions

Some questions that will be answered over the next few sections:

* What is an embedding?
* What is a good representation?
* Why are embeddings considered good representations?

## Representations

It's often said that deep learning is about learning good representations. In traditional machine learning fields, representations were often concocted by hand by domain experts and researchers. **We'll see later why good representations matter, and what makes a representation good.**

Say you have a million words in your vocabulary and a machine learning model has one of those words as an input. How would you represent the word as a vector?

## One-Hot Vectors

Traditionally, you would have a vector with 1 million components, or dimensions, and you would set the value of every dimension to $0$ except the dimension corresponding to the word in question (based on an index), which you would set to one. We call this **one-hot**.

Let's be *formal* and use some mathematical notation:

*Formally*, the one-hot vector for the word of index $i$:

$$ x = \begin{bmatrix} 0 & 0 & ... & 1  & ... & 0  \end{bmatrix} $$

Where the $i^{th}$ element is 1 and all other elements are 0.

## Dense representation

In contrast to one-hot vectors, which are sparse (contain many 0s), Word2Vec trains dense vectors, which contain no 0s and are of lower dimension (say 100 - 300), significantly smaller than the vocabulary. The vector for every word is different; every word vector points in a different direction. These dense vectors are collected in column form to make an **embedding matrix**.

### Formulation

*Formally*

The **dense representation** of a one-hot encoded word $x$ that represents the $c^{th}$ word in the vocabulary is:

$$ V x =  \vec v_c $$

Where $V$ is the embedding matrix

And $\vec v_c$ is a vector in embedding space

### Explanation

This looks like matrix-vector multiplication, but it's actually even simpler. If you remember you matrix-vector multiplication rules, a matrix $V$ times a vector $x$ is a linear combination of all columns in matrix $V$ by every element of vector $x$. In other words, for every column in $V$, multiply the column vector $\vec v_i$ by the scalar $x_i$ and add up all $v_i$s.

$$ Vx = \sum_i^{|W|} x_i \vec v_i $$

Since $x$ is one-hot, only one column is multiplied by a non-zero element. This is actually just selecting column $i$ from matrix $V$. $V$ is *precisely* a collection of word vectors in column form.

### Examples

To embed the one-hot vector $x$ for word $c$ from embedding matrix $V$:

$$ Vx = \vec v_c $$

To embed the one-hot vector $y$ for word $o$ from embedding matrix $U$:

$$ Uy = \vec u_o $$

## Answers

What is an embedding?

** A dense vector representation of a word **

** A matrix with $|W|$ columns of dense vectors **

## Vector Space Model

We have a collection of word vectors for words in our vocabulary - how do we know if they are good representations?

It must be the case that a **good word representation** have certain properties. Let's first look at what those properties are and later on demonstrate how those properties can be achieved.

Recall from linear algebra that two vectors of the same dimension can be combined in two ways:

### Vector Sum

The sum of vectors $v$ and $u$ is defined as addition of their individual elements.

$$ w_i = v_i + u_i $$

The resulting vector is the equivalent of adding drawing the tail of $u$ from the head of $v$:

### Dot Product

The dot product of vectors $\vec v_c$ and $\vec u_o$ is defined as the sum of the product of their individual elements.

$$ \vec v_c^T \vec u_o = \sum_i v_i * u_i $$

#### Similarity

The dot product is proportional to the similarity of 2 vectors:

Vectors that point in the same direction have a high dot product.
Vectors that point in opposite directions have a low dot product.
Vectors that point in unrelated (orthogonal) directions have 0 dot product.

### Word2Vec vs One-Hot

One-hot vectors have 0 dot product. This implies that all words are equally unrelated to each other. ** This is neither true nor desirable **

Dense embeddings have high dot products for words that appear together and low dot products for words that do not. If we believe that similar & related words appear together, then we'd like them to point in similar directions and thus have a high dot product.

### Good Representation

**A good representation encodes as much relevant information as possible. It accurately depicts prior knowledge we have about the data, and helps us create a model for the underlying task that takes advantage of that prior knowledge.**

For example, with a task like question-answering, we'd like to not have to learn word relationships from scratch. When we have access to a large dataset of words that we can train embeddings on, but a comparatively smaller dataset to train on our task, we'd like our task to use as much outside knowledge as possible.

We know that words are related to each other in along different axis.

#### Examples

* The relationship between "king" and "man" is along the axis of instantiation - a king is an instance of a man
* The relationship between "man" and "woman" is along the axis of gender
* The relationship between "one" and "two" is along the axis of plurality - two is a multiple of one

Dense embeddings represent the relatedness of words, whereas one-hot encodings do not. A downstream machine learning task can make use of this insight and will not have to re-learn how "man" and "king" are related. When a good representation is learned, the overall task is becomes easier.

#### Parallel Relationships

![](https://www.tensorflow.org/versions/r0.11/images/linear-relationships.png)

$\vec{man}$ and $\vec{woman}$ line up with each other in the same way that $\vec{king}$ and $\vec{queen}$ do

## Answers

What is a good representation?

** Good representations take advantage of prior knowledge to make a downstream machine learning task easier **

Why are dense embeddings good representations?

** Dense embeddings represent words based on their relatedness, so that a downstream task can make use of such relatedness without having to learn it on its own. The dot product of two related words is high, and the dot product of two unrelated words is low. **

# The Word2Vec Model

** You shall know a word by its neighbors. **

Word2Vec trains a probabilistlic model to maximize the probability of seeing a neighbor word given an input word from a given training corpus.

## Questions

* What is the Word2Vec model trained on?
* How does Word2Vec update individual word vectors when training?
* Why do those updates lead to vector space properties discussed above?

## From Vector Space Model to Probabilistic Model

We talked about word vectors, their dot products, and observed certain properties. How do we get those properties?


### The Data

"Berlin is a city in East Germany"

Let's say our dataset is Wikipedia, which contains many sentences like the one above. Notice that the words in the sentence relate to one another in some way, except for somewhat meaningless words like "is" and "a."

#### Ponder the relatedness

* Consider how the words "Berlin" and "Germany" might appear together in the same way that "Paris" and "France" do
* Consider how the words "Berlin" and "city" appear together the same way "Paris" and "city" do

### The Setup

Let's call "Berlin is a city in East Germany" the current context.

Let's call the input word "city" because it is in the center of the context, and let's call it the "center" word for additional clarity.

"Berlin" "is" "a" "in" "East" "Germany" are all neighbor words.

For every neighbor word, we have an input-output training pair, like so:

"city" -> "Berlin"

"city" -> "is"

"city" -> "a"

"city" -> "in"

"city" -> "East"

"city" -> "Germany"

### The Objective

The objective function is the maximize the probability of the output word given the input word for every input-output training pair.

**Formally**
$o,c$ is a training pair

$ \forall {o,c} $ maximize:

$$ p(o | c) $$

#### Conditional Probability

Recall from probability theory that the conditional probability

$$ p(o | c) = \frac {p(o, c)}{p(c)} $$

Notice that the numerator is the joint probability. What happens when you increase the joint probability and hold $p(c)$ fixed? The conditional probability also goes up.

** Let's relate this back to word vectors **

#### Word Vectors and Scores

Recall that we had a word embedding, which we conveniently called $\vec v_c$.

Recall that the similarity between two word vectors was the dot product. I'm going to call that dot product the **score** or similarity, and call the function $s(\cdot)$. (see appendix for more detail and characterization).

**NOW** consider that we have two embedding matrices, $V$ and $U$, where $V$ is used for the "center" word $c$ and $U$ is used for the output word $o$. (The justification for using two embeddings is outside the scope of these notes)

The score is defined as:

$$ s(o,c) = \vec v_c^T \vec u_o$$

What is desirable in our model?

* We want the score for center word $c$ and output word $o$ to be high
* We want the score for center word $c$ and all other words
* We want to output probability

Let's use this last point as the motivator to define the final form of the model we train.

#### Back to probability-land

Probabilities must be positive and sum to 1. To achieve the former, we use can use the exponential function:

$$e^x$$

Which we just write as $exp(\cdot)$ for readability.

The latter is achieved with the denominator of the conditional probability formula, which in exponential form is:

$$\sum_w exp(x_w)$$

over $w$ classes.

You might recognize this as the softmax function, because it is!

$$\frac{exp(x_i)}{\sum_w exp(x_w)} $$

In this model, I set $x$ to the score, and I approximate the unnormalized joint probability like this:

$$ p(o,c) \approx exp(s(o,c)) $$

And the total probability:

$$ p(c) \approx \sum_{w}{exp(s(w,c))} $$

For all $w$ words in the vocabulary, including $o$.

Then:

$$ p(o|c) \approx \frac{exp(s(o,c))}{\sum_w{exp(s(w,c))}} = \frac{\exp(\vec v_c^T \vec u_o)}{\sum_w{\exp(\vec v_c^T \vec u_w)}} $$

What a handful!

Put another way, we want to make the probability high for the output word $o$ and low for all other $w$ words, which amounts to making the score high for $o$ and low for all other $w$.

### 

Different words occur together in different contexts. When we model words in *vector space*, we make the vectors point in similar directions based on how those words occur together. Vectors with similar components point in similar directions, and have high dot products. Vectors with similar direction have similar neighbors.

We'd like the model to output high probabilities for words that occur often together, and low probabilities for words that don't. The probability of seeing the word $Germany$ near the word $Berlin$ should be high, whereas the probability of seeing the word $potato$ near the word $Berlin$ should be low.

Finally, $Berlin$ and $Germany$ appear near words like $city$ and $capitol$, just like $Paris$ and $France$ might. More on this example later.


## Answers

What is Word2Vec trained on?

** Word2Vec is trained on input-output pairs of words, where the output word is a neighbor of the input word in the training corpus **

What is the objective function of Word2Vec?

** Word2Vec maximizes the conditional probability of the output word given the input word. This is equivalent to maximizing the joint probability of seeing the input and output word together. **

## Training

We have our model:

$$ \frac{\exp(\vec v_c^T \vec u_o)}{\sum_w{\exp(\vec v_c^T \vec u_w)}} $$

There are not one, but **two** sets of word vectors to optimize. When we're done, we can combine them in some arbitrary way (let's say sum). As in turns out, these vectors will be learning pretty similar things, and it was a matter of experimentation that the creators of word2vec ended up with 2 sets.

The word vectors to update are $v_c$ for the center word and $U$ for all possible output words, including the "correct" output word $o$.

### Initialization

We have no concept or preference of how these word vectors might look like, so we rely entirely on the data. Let's just initialize these vectors to a bunch of random small numbers.

### Gradients

There are entire courses on optimization of neural networks; instead of going over them in any amount of detail and not doing them justice, I'm going to tell you that they are black boxes and just give you the "gradients" that you will be using.

The gradients for each word vector tell you how to update them for every example you see. This process is called Stochastic Gradient Descent, and the gradients come from applying the chain rule of calculus in a neural network using the process of Backpropagataion.

Something to pay attention to:
Signs are important! I subtract the gradient. When I add a gradient to a vector, I'm pushing that vector towards the gradient. (show diagram)

### Center word

Let's look at the gradient for $v_c$, which we call $\frac{\partial}{\partial v_c}$, and see if we can gain some intuition about it.

#### Gradient Update

You are given the center word and need to update the word vector upon seeing a new example. The example tells you that the center word and output word should have a higher score together. It also implies that the center word and all other words should have a lower score. Let's see how that plays out in the math.

Let's use $Berlin$ - $city$ and $Germany$ - $city$ as an example. By moving the vectors of $Berlin$ and $city$ closer together, as well as $Germany$ and $city$, we get vectors that are similar

$$ \frac{\partial}{\partial v_c} = \sum_w^W p(w|c) \vec u_w - \vec u_o = \sum_{w \ne o}^{W} p(w|c) * u_{i,j}  -  (p(o|c) - 1) u_o $$

$$ v_c = v_c - \alpha * \frac{\partial}{\partial v_c} $$

#### Vector Space Intuition

What is this update doing? It's adding a small multiple $\alpha$ of the gradient, which we can further break down.

First, it's adding a small bit of $u_o$. What happens when you add $u_o$ to $v_c$?

$v_c$ moves towards $u_o$ - it becomes more similar to $u_o$. This makes sense, because we're trying to increase their dot product and therefore their similarity.

Now, let's add the sum of all $u_w$ vectors. Individually, what is happening? You're adding the negative of every $u_w$ vector. $v_c$ moves away from every $u_w$, becoming less similar. This is because we're trying to decrease their dot product.

### Output words

Let's look at the gradient for $u_o$ and all other $u_w$s and see if we can gain some intution:

$$
\begin{equation}
    \frac{\partial}{\partial u_w} =
    \begin{cases}
    \hat y_w \cdot v_c - v_c = v_c(\hat y_w - 1) &\text{if } w = o \\[2ex]
    \hat y_w \cdot v_c = v_c * \hat y_w &\text{if } w \ne o
    \end{cases}
\end{equation}
$$

$$ u_w = u_w - \alpha * \frac{\partial}{\partial u_w} $$

#### Vector Space Intuition

What is this update doing?

For the correct output word $u_o$, it's adding a small multiple $\alpha$ times the center word vector $v_c$, scaled by the probability error (how much it is less than 1). It's moving $u_o$ slightly towards $v_c$.

For all other output words $u_w$, it's subtracting a small multiple times the center word vector $v_c$, scaled by the predicted probability (how much it is greater than 0). It's moving $u_w$ slightly away from $v_c$.


### Illusion, Explained

When you encounter a pair of words, they are treated as related and their word vectors move towards each other. Various pairs of words push and pull, and they start to line up in ways that we perceive to be their relationships.

This is merely an illusion of the words being neighbors in different contexts. Similar pairs of co-occurrences, i.e. $Paris$ - $France$ and $Berlin$ - $Germany$, end up resulting in similar movements.

Consider updates from repeated examples of $Paris$ - $capitol$ and $Berlin$ - $capitol$, as well as pairs such as $Paris$ - $Berlin$ and $France$ - $Germany$. $Paris$ and $Berlin$ are pulled towards $capitol$, as well as $France$ and $Germany$. This causes the $Paris$ - $Berlin$ pair to become parallel to the $France$ - $Germany$ pair.

That's how you're able to do analogical arithmetic. Magic no more!

### Answer

How does Word2Vec update individual word vectors when training?

** Word2Vec moves the vectors for the center and output word towards each other by adding a small multiple of each one to the other **

Why do those updates lead to vector space properties discussed above?

** Words with similar neighbors move together in parallel. This parallel movement results in arrangements that are where related word vector pairs form diamond shapes. **

# Applications & Appendix

## Conceptual Application

When you have complicated models, you have more transformations between dense representations.
However, the intuition remains the same; you're dealing with dot products and similarities (negative distances).

As you train models with either embeddings or hidden layers, keep in mind that you're always either trying to increase or decrease dot products between your weights and your examples. In a sense, you are memorizing your dataset in a continuous way. With enough weights, it's no surprise that you can overfit!

## Usage

In a deep learning language model, you can "pretrain" your word vectors using Word2Vec to use as inputs into your first layer of a recurrent neural network. You can also keep the weights in your network and "finetune" them by backpropagating into them.

You can also use the $V$ matrix as a "de-embedding" matrix for going from the final hidden state (which could be recurrent) of your model to scores over your vocabulary, which you would run a softmax over.


## Appendix

### Unnormalized Log Probability

$ p(o,c) $ is the joint probability, and it is proportional to:

$ \vec v_c^T \vec u_o$, which is the dot product.

If we exponentiate the dot product

$exp(\vec v_c^T \vec u_o)$

Then normalize by all exponentiated dot products over $c$

$\sum_w exp(\vec v_c^T \vec u_w)$

That starts to look like $\frac{p(o,c)}{p(c)}$, which is in fact the probability we were after.

To summarize:

* We exponentiated the dot product
* We normalized over all exponentiated dot products for a center word
* In the inverse, we can say we got to the dot product from the probability by unnormalizing and taking the log.

### Cross Entropy & Derivation

#### Definition

Cross entropy is an error measure that represents the different between a predicted probability distribution and the true distribution.

$CE(y, \hat y) = - \sum_i y_i log(\hat y_i)$

#### Derivation of cross entropy gradient

$CE(y, \hat y) = - \sum_i y_i log(\hat y_i)$

$ \hat y_i = \frac{e^{x_i}}{\sum_j e^{x_j}} $

$\frac{\partial}{\partial x_k} CE(y, \hat y) = 
- \frac{\partial}{\partial x_k} \sum_i y_i log(\hat y_i)$

$= - \frac{\partial}{\partial x_k} log(\hat y_i)$

$= - \frac{\partial}{\partial x_k} log \frac{e^{x_i}}{\sum_j e^{x_j}} $

$= - ( \frac{\partial}{\partial x_k} \log e^{x_i} - \frac{\partial}{\partial x_k} \log \sum_j e^{x_j} ) $

$= - ( \frac{\partial}{\partial x_k} x_i - \frac{\partial}{\partial x_k} \log \sum_j e^{x_j} ) $

$= - ( \frac{\partial}{\partial x_k} x_i - \frac{\partial}{\partial x_k} \log \sum_j e^{x_j} ) $

$= - ( \frac{\partial}{\partial x_k} x_i - \frac{1}{\sum_j e^{x_j}} * \frac{\partial}{\partial x_k} \sum_j e^{x_j} ) $

$= - ( \frac{\partial}{\partial x_k} x_i - \frac{1}{\sum_j e^{x_j}} * \frac{\partial}{\partial x_k} e^{x_k} ) $

$= - ( \frac{\partial}{\partial x_k} x_i - \frac{e^{x_k}}{\sum_j e^{x_j}}) $

$= - ( \frac{\partial}{\partial x_k} x_i - \hat y_k) $

$= \hat y_k - \frac{\partial}{\partial x_k} x_i $

$$
\begin{equation}
    \frac{\partial}{\partial x_k} CE(y, \hat y) =
    \begin{cases}
    \hat y_k - 1 &\text{if } i = k \\[2ex]
    \hat y_k - 0 &\text{if } i \ne k
    \end{cases}
\end{equation}
$$

OR

$\frac{\partial}{\partial x} CE(y, \hat y) = \hat y - y$

### Word2Vec Derivation

#### Derive gradients wrt $v_c$

$J_{softmax-CE}(o, v_c, u_o) = - \sum_i y_i \log \hat y_i = - \log \hat y_y$

where $\hat y_y$ is the predicted probability for the correct output word $y$.

$$ \hat y_o = p(o | c) $$

For an individual outer-word + inner-word pair $o, c$, the `score`, or unnormalized log probability is defined as:

$$z_{w,c} = u_w^T v_c$$

You can vectorize the `score` by thinking of it as a matrix-vector product:

$z_c = U v_c$

Where $U$ is a row-matrix of outer-word vectors. The `score` represents the inner-product similarity between the two words, and the softmax function will accentuate the highest scoring pair.

$$softmax(z_{o,c}) = \frac {exp(z_{o,c})}{\sum_{w=1}{W} exp(z_{w,c})}$$

$$\approx \max_w z_c$$

Since $y_i = 0 \forall i \ne k$

On individual elements:

$$\frac {\partial}{\partial v_c} - \log \hat y$$

$$= \frac {\partial}{\partial v_c} - \log softmax(z_{o,c})$$

$$= \frac {\partial}{\partial v_c} - \log \frac {\exp(z_{o,c})}{\sum_{w=1}{W} (u_w^T v_c)}$$

$$= - [\frac {\partial}{\partial v_c} \log \frac {\exp(z_{o,c})}{\sum_{w=1}{W} (u_w^T v_c)}]$$

$$= - [\frac {\partial}{\partial v_c} \log exp(z_{o,c}) - \frac {\partial}{\partial v_c} \log \sum_{w=1}^W (u_w^T v_c)]$$

$$= \frac {\partial}{\partial v_c} \log \sum_{w=1}^W \exp(u_w^T v_c) - \frac {\partial}{\partial v_c} \log exp(z_{o,c})$$

$$= \frac {\partial}{\partial v_c} \log \sum_{w=1}^W \exp(u_w^T v_c) - \frac {\partial}{\partial v_c} z_{o,c}$$

$$= \frac {\partial}{\partial v_c} \log \sum_{w=1}^W \exp(u_w^T v_c) - \frac {\partial}{\partial v_c} z_{o,c}$$

$$= \frac {1}{\sum_{w=1}^W \exp(u_w^T v_c)} \frac {\partial}{\partial v_c} \sum_{x=1}^W \exp(u_x^T v_c) - \frac {\partial}{\partial v_c} z_{o,c}$$

$$= \sum_{x=1}^W \frac {1}{\sum_{w=1}^W \exp(u_w^T v_c)}  \frac {\partial}{\partial v_c} \exp(u_x^T v_c) - \frac {\partial}{\partial v_c} z_{o,c}$$

$$= \sum_{x=1}^W \frac {1}{\sum_{w=1}^W \exp(u_w^T v_c)} \exp(u_x^T v_c) \cdot u_x - \frac {\partial}{\partial v_c} z_{o,c}$$

$$= \sum_{x=1}^W \frac {\exp(u_x^T v_c)}{\sum_{w=1}^W \exp(u_w^T v_c)}  \cdot u_x - \frac {\partial}{\partial v_c} z_{o,c}$$

$$= \sum_{x=1}^W p(x|c) \cdot u_x - \frac {\partial}{\partial v_c} z_{o,c}$$

$$= \sum_{x=1}^W \hat y_x \cdot u_x - \frac {\partial}{\partial v_c} u_o^T v_c$$

$$= \sum_{x=1}^W \hat y_x \cdot u_x - u_o$$

Equivalently, on vectors:

$\frac{\partial}{\partial v_c} J_{softmax-CE}(o, v_c, U)$

$$= \frac{\partial J}{\partial z_c} \frac{\partial z_c}{\partial v_c}$$

Because $\frac{\partial CE(y, \hat y)}{\partial \theta} = \hat y - y$ from above:

$$= (\hat y - y) ^T \frac{\partial z_c}{\partial v_c}$$ (to match dimensions)

Because $z_c = U v_c$ and the gradient is the transpose of the Jacobian:

$$= ((\hat y - y) U)^T$$

$$= U^T (\hat y - y)$$

#### Derive gradients wrt all $u_w$ including $u_o$

As before:

$J_{softmax-CE}(c, v_c, u_o) = - \sum_i y_i \log \hat y = - \log \hat y$

Since $y_i = 0 \forall i \ne k$

$\hat y_o = p(o | c)$

$z_{w,c} = u_w^T v_c$

$softmax(z_{o,c}) = \frac {exp(z_{o,c})}{\sum_{w=1}{W} (u_w^T v_c)}$

On individual elements:

$$\frac {\partial}{\partial u_w} - \log \hat y$$

$$= \frac {\partial}{\partial u_w} - \log softmax(z_{o,c})$$

$$= \frac {\partial}{\partial u_w} - \log \frac {\exp(z_{o,c})}{\sum_{w=1}{W} (u_w^T v_c)}$$

$$= - [\frac {\partial}{\partial u_w} \log \frac {\exp(z_{o,c})}{\sum_{w=1}{W} (u_w^T v_c)}]$$

$$= - [\frac {\partial}{\partial u_w} \log exp(z_{o,c}) - \frac {\partial}{\partial v_c} \log \sum_{w=1}^W (u_w^T v_c)]$$

$$= \frac {\partial}{\partial u_w} \log \sum_{w=1}^W \exp(u_w^T v_c) - \frac {\partial}{\partial u_w} \log exp(z_{o,c})$$

$$= \frac {\partial}{\partial u_w} \log \sum_{w=1}^W \exp(u_w^T v_c) - \frac {\partial}{\partial u_w} z_{o,c}$$

$$= \frac {1}{\sum_{w=1}^W \exp(u_w^T v_c)} \frac {\partial}{\partial u_w} \sum_{x=1}^W \exp(u_x^T v_c) - \frac {\partial}{\partial u_w} z_{o,c}$$

The sum goes away since the derivative is 0 when $x \ne w$

$$= \frac {1}{\sum_{w=1}^W \exp(u_w^T v_c)} \frac {\partial}{\partial u_w} \exp(u_w^T v_c) - \frac {\partial}{\partial u_w} z_{o,c}$$

$$= \frac {1}{\sum_{w=1}^W \exp(u_w^T v_c)} \exp(u_w^T v_c) \frac {\partial}{\partial u_w} u_w^T v_c - \frac {\partial}{\partial u_w} z_{o,c}$$

$$= \frac {1}{\sum_{w=1}^W \exp(u_w^T v_c)} \exp(u_w^T v_c) \cdot v_c - \frac {\partial}{\partial u_w} z_{o,c}$$

$$= \frac {\exp(u_w^T v_c)}{\sum_{w=1}^W \exp(u_w^T v_c)} \cdot v_c - \frac {\partial}{\partial u_w} z_{o,c}$$

$$= p(w | c) \cdot v_c - \frac {\partial}{\partial u_w} z_{o,c}$$

$$= \hat y_w \cdot v_c - \frac {\partial}{\partial u_w} z_{o,c}$$

$$= \hat y_w \cdot v_c - \frac {\partial}{\partial u_w} u_o^T v_c$$

$$
\begin{equation}
    \frac{\partial}{\partial u_w} CE(y, \hat y) =
    \begin{cases}
    \hat y_w \cdot v_c - v_c = v_c(\hat y_w - 1) &\text{if } w = o \\[2ex]
    \hat y_w \cdot v_c = v_c * \hat y_w &\text{if } w \ne o
    \end{cases}
\end{equation}
$$


### Euclidian Word Distance

L2 distance between 2 word vectors 
$$ d^2 = (x - z)^2 = x^2 + z^2 - 2xz $$

Setting them to normal vectors
$$ d^2 = 2 - 2xz $$

Then exponentiating for unnormalized probability ->

$$ exp(-d^2) $$


