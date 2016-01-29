---
layout: post
title: 'Gated Feedback Recurrent Networks'
date: 2016-01-28 16:41
comments: true
categories:
---

## tl;dr

Gated Feedback Recurrent Networks are a nice improvement to traditional RNNs and their popular variants, the LSTM and GRU. The architectural change is minimal, and the implementation in Tensorflow follows closely.

## Intro

Our deep learning study group dove into a detailed study of RNN architecture, focusing on LSTMs. We read the LSTM Search paper (http://arxiv.org/pdf/1503.04069.pdf), and one of our members did a [great job](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93) of implementing the variants in Tensorflow and recreating the performance studies. Next, we read a paper on a new variant of RNNs designed to overcome the problem of sequences that operate at different time lags. The paper introduces the concept of a Gated Feedback Recurrent Neural Network, with the operative change being the additional 'feedback' connections that are themselves gated, using the same principles underlying the LSTM and GRU.

## Recurrent Neural Networks

Recurrent Neural Networks (RNNs) present a flexible API for modeling input and output sequences of arbitrary length, in contrast to n-gram models and HMMs. RNNs try to model the underlying pattern in the sequence via the hidden state, with each hidden state depends on the hidden state at the previous timestep and the input at the current timestep. The parameters at all timesteps are tied, so the weights reflect the structure across time.

In a single layer definition (easily generalizable to multiple layers):

$$
h_t = tanh(W x_t + U h_{t - 1} + b)
$$

$$
y_t = softmax(h_t)
$$

In the above equations:

* $W$, $U$ and $b$ are weights and biases for applying an affine transformation of the inputs $x_t$ and previous hidden state $h_{t-1}$
* The hidden state, $h_t$, is also the output, which is run through a softmax layer to output probabilities

The vanilla RNN formulation has trouble capturing long-term dependencies due to the vanishing gradient problem. In practice, backpropagation through time tends to lose information after many timesteps.

## LSTM / GRU

The Long Short Term Memory network (LSTM), solves the vanishing gradient problem by introducing a memory cell that is updated conditionally. Influence over and from this cell is controlled by Gates; nonlinear, differentiable transformations that are governed by learnable parameters and that control the flow of data. Over the course of several epochs worth of examples, the gates learn when to open and close, letting information flow in and out of the cell accordingly. The method was [first introduced in 1997](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) and its variants were explored throughout the years.

A recent, drastic simplification of the LSTM is the Gated Recurrent Unit (GRU), which merges functionality of the cell and the hidden state. Studies demonstrate competitive performance, especially accounting for the reduced parameter space of the GRU for a hidden state of the same size (due to not having a separate cell state and accompanying gate parameters). A wonderful and intuitive summary can be found on this [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

## Gated Feedback Recurrent Networks





## Implementation

### Simplified

Made assumption based on modifying formula. Try to avoid leakage of Gated Feedback abstraction. Critique round with study group. Email exchange with lead on paper.

### Corrected

Inject gated feedback into implementation. Use good subclassing.


### Tensorflow Canonical

Minimize operations through vectorization, TF style.

## Contribution on GitHub

Demo runs with inline implementation can be found [here](https://github.com/ZhangBanger/highway-vs-gradnet)
