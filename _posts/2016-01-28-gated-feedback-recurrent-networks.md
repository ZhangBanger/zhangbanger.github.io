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
* Notice that $W$, $U$, and $b$ are not indexed by timestep. This is because the **weights are tied, or shared, across all of time.**

In very explicit Tensorflow code, we reuse symbolic variables `W`, `U`, and `b` across different timesteps, making sure they are under the same variable scope in any given layer. 

```python
with tf.variable_scope(scope or type(self).__name__):
            x = inputs
            h_t_1 = state
            W = tf.get_variable("W", [self.input_size, self.state_size],
                                initializer=tf.random_uniform_initializer(-0.1, 0.1))
            U = tf.get_variable("U", [self.state_size, self.state_size],
                                initializer=tf.random_uniform_initializer(-0.1, 0.1))
            b = tf.get_variable("b", [self.state_size], tf.constant_initializer(0.0))
            h_t = tf.tanh(tf.matmul(x, W) + tf.matmul(h_t_1, U) + b)

        return h_t, h_t
```

The actual TF implementation further combines matrix operations for efficiency:

Code:

```python
output = tanh(linear([inputs, state], self._num_units, True))
```

Math:
$$
h_t = tanh(W \cdot [x_t, h_{t - 1}] + b)
$$

`linear` is the utility function that concatenates `inputs` and `state` and applies a single, larger `Wx + b`.

Conceptually, RNNs apply the same function $A$ to different inputs and hidden states at every point in time. The realization can be **unrolled** in such a way:

[![Unrolled RNN][1]][1]
[1]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png

[Source](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Sidenote: In a sense, if a neural network is like a function, a recurrent neural network is like a small program.

The vanilla RNN formulation has trouble capturing long-term dependencies due to the vanishing gradient problem. In practice, backpropagation through time tends to lose information after many timesteps.

## LSTM

The Long Short Term Memory network (LSTM), solves the vanishing gradient problem by introducing a memory cell that is updated conditionally. Influence over and from this cell is controlled by Gates; nonlinear, differentiable transformations that are governed by learnable parameters and that control the flow of data. Over the course of several epochs worth of examples, the gates learn when to open and close, letting information flow in and out of the cell accordingly. The method was [first introduced in 1997](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) and its variants were explored throughout the years.

For a given layer, the new cell content $c_t$ is calculated as:

$$
c_t = f_t c_{t - 1} + i_t \tilde c_t
$$

It is a combination of the old cell content gated by a 'forget gate' $f_t$ and the candidate cell content gated by an 'input gate' $i_t$.

The output of the unit is equal to the new hidden state, which applies an output gate to the new cell state:

$$
h_t = o_t \cdot \tanh c_t
$$

The candidate cell content is calculated purely from the current input and previous hidden state. In other words, the old cell content is never directly leaked into it:

$$
\tilde c_t = \tanh (W_c x_t + U_c h_{t - 1})
$$

The forget gate, input gate, and output gate are all derived from the current input and previous hidden state:

$$
f_t = \sigma(W_f x_f + U_f h_{t - 1} + b_f)
$$

$$
i_t = \sigma(W_i x_t + U_i h_{t - 1} + b_i)
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t - 1} + b_o)
$$

The gates are dot products, and learn to open and close for every component in the vector based on the internal representation of the sequence (the hidden state), which is **different** from the internal memory of the sequence (the cell state).

Tensorflow's verbose vs optimized implementations resemble the GRU ones, described in the next section.

## GRU

A recent, drastic simplification of the LSTM is the Gated Recurrent Unit (GRU), which merges functionality of the cell and the hidden state. Studies demonstrate competitive performance, especially accounting for the reduced parameter space of the GRU for a hidden state of the same size (due to not having a separate cell state and accompanying gate parameters). A wonderful and intuitive summary can be found on this [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

The hidden state update resembles the cell state update:

$$
h_t = (1 - z_t) h_{t - 1} + z_t \tilde h_t
$$

The update gate merges the functionality of input and forget by making one the additive inverse of the other. It's formulation is otherwise the same:

$$
z_t = \sigma(W_z x_t + U_z h_{t - 1} + b_z)
$$

However, the way a candidate is created incorporates an additional 'reset' gate:

$$
\tilde h_t = \tanh (W x_t + r_t \odot U h_{t - 1})
$$

where $\odot$ is pointwise multiplication of 2 vectors rather than a dot product.

The reset gate is compute in a similar way as other gates we've seen:

$$
r_t = \sigma (W_r x_t + U_r h_{t - 1} + b_r)
$$

The reset gate controls the mix of the previous hidden state that is incorporated in the candidate hidden state.

This combination of gates is a different way to accomplish what the LSTM does so well - model the beginning and end of arbitrarily long sequences.

In code:

```python
with tf.variable_scope(scope or type(self).__name__):
    h_t_prev, _ = tf.split(1, 2, state)
    x_t = inputs
    with tf.variable_scope("Update Gate"):
        W_z = tf.get_variable("W_z", [self.input_size, self._num_units],
                              initializer=tf.random_uniform_initializer(-0.1, 0.1))
        U_z = tf.get_variable("U_z", [self.input_size, self._num_units],
                              initializer=tf.random_uniform_initializer(-0.1, 0.1))
        b_z = tf.get_variable("b_z", [self._num_units], tf.constant_initializer(0.0))

        z_t = tf.sigmoid(tf.matmul(x_t, W_z) + tf.matmul(h_t_prev, U_z) + b_z, name="z_t")

    with tf.variable_scope("Reset Gate"):
        W_r = tf.get_variable("W_r", [self.input_size, self._num_units],
                              initializer=tf.random_uniform_initializer(-0.1, 0.1))
        U_r = tf.get_variable("U_r", [self.input_size, self._num_units],
                              initializer=tf.random_uniform_initializer(-0.1, 0.1))
        b_r = tf.get_variable("b_r", [self._num_units], tf.constant_initializer(1.0))

        r_t = tf.sigmoid(tf.matmul(x_t, W_r) + tf.matmul(h_t_prev, U_r) + b_r, name="r_t")

    with tf.variable_scope("Candidate"):
        # New memory content
        W = tf.get_variable("W", [self.input_size, self._num_units],
                            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        U = tf.get_variable("U", [self.input_size, self._num_units],
                            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        b = tf.get_variable("b", [self._num_units], tf.constant_initializer(0.0))
        hc_t = tf.tanh(tf.matmul(x_t, W) + tf.mul(r_t, tf.matmul(h_t_prev, U) + b))

    with tf.Variable("Output"):
        h_t = tf.mul(z_t, hc_t) + tf.mul((1 - z_t), h_t_prev)

return h_t, h_t
```

You can construct a more compact mathematical in the same way we did above. In fact, you can also combine the gate calculations into 1 matrix multiply:

```python
with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
    with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not udpate.
        r, u = array_ops.split(1, 2, linear([inputs, state],
                                        2 * self._num_units, True, 1.0))
        r, u = sigmoid(r), sigmoid(u)
    with vs.variable_scope("Candidate"):
        c = tanh(linear([inputs, r * state], self._num_units, True))
    new_h = u * state + (1 - u) * c
    return new_h, new_h
```
[Source](https://github.com/tensorflow/tensorflow/blob/182bc43407d24ef9d1cd44f70726a847731599d9/tensorflow/python/ops/rnn_cell.py#L144-L155)

## Gated Feedback Recurrent Networks

We've solved the vanishing gradient problem with gating, so we can deal with longer timescale dependencies. Past approaches involved stacking multiple layers of RNN cells, but those tended to learn abstractions rather than timescale independence. A heavily-engineered approach called the [ClockWork-RNN](http://arxiv.org/pdf/1402.3511v1.pdf) forces the different layers to operate at different timescales, but the timescales themselves were tuned rather than learned.

The [paper we read](http://arxiv.org/pdf/1502.02367v4.pdf) introduces the Gated Feedback Recurrent Neural Network (GF-RNN), where timescale invariance is achieved by creating direct connections for signals between all layers of the previous timestep. Effectively, each timestep becomes "fully-connected" to the previous. It's not intuitively obvious to me how this works, but some examples in the paper show variable-length dependencies that were better handled by the GF-RNN.

To control direct connections with previous hidden states, global reset gates are used:

$$
g^{i \to j} = \sigma (w_g^{i \to j} h_t^{j - 1} + u_g^{i \to j} h_{t - 1}^*)
$$

$g^{i \to j}$ is the global reset scalar value that is separately learned for every hidden layer $i$ in the previous timestep in order to calculate every hidden layer $j$ in the current timestep.
$w_g^{i \to j}$ and $u_g^{i \to j}$ are weight vectors for the input and concatenation of the previous hidden states $h_{t - 1}^*$.

In the case of a GRU, these connections are injected into candidate generation. Rather than applying the (now) local reset gate to just the previous hidden state at the current layer to obtain the contribution from the previous timestep, the local reset is applied to a sum of all hidden states in all layers in previous timestep, each with a weight matrix and global reset scalar.

$$
\tilde h_t^j = tanh(W h_t^{j - 1} + r_t^j \odot \sum_{i = 1}^L g^{i \to j} U^{i \to j} h_{t - 1}^i)
$$

Notice that the term on the left doesn't change.

## Implementation

### Simplified

I attempted a simplified implementation where I actually modified the formula a bit. I wanted to avoid leaking the Gated Feedback abstraction into the `RNNCell` layer, instead handling it at the `MultiRNNCell` layer, which is a class that stacks `RNNCell`s for you and handles passing information between layers.

The right term in $\tilde h_t^j$ would remain unchanged from the original GRU definition, and instead:

$$
h_{t - 1} \leftarrow \sum_{i = 1}^L g^{i \to j} h_{t - 1}^i
$$

But in fact, $g^{i \to j}$ becomes $g^j$ because the term no longer is depends on $i$ and is calculated outside of the cell altogether.

In code, the change from `MultiRNNCell` is under the "Global Reset" scope:

```python
with tf.variable_scope(scope or type(self).__name__):
    # Conveniently the concatenation of all hidden states at t-1
    h_star_t_prev = state
    cur_state_pos = 0
    cur_inp = inputs
    new_states = []
    for i, cell in enumerate(self._cells):
        with tf.variable_scope("Cell%d" % i):
            cur_state = array_ops.slice(
                    state, [0, cur_state_pos], [-1, cell.state_size])
            with tf.variable_scope("Global Reset"):
                u_g = tf.get_variable("u_g", [self.state_size],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                w_g = tf.get_variable("w_g", cell.state_size,
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                g = tf.sigmoid(tf.reduce_sum(tf.mul(w_g, cur_inp)) + tf.reduce_sum(tf.mul(u_g, h_star_t_prev)))
                cur_state = tf.reduce_sum(g * cur_state)

            cur_state_pos += cell.state_size
            cur_inp, new_state = cell(cur_inp, cur_state)
            new_states.append(new_state)

return cur_inp, array_ops.concat(1, new_states)
```
No change was needed for `GRUCell`.

When I brought this to the study group, the flaws of this approach became apparant. Not having separate gates for each $i \to j$ pair didn't provide enough degrees of freedom for the gate.
Also, the paper's stance on only applying the global reset to the candidate generation was confirmed by a quick email exchange with Junyoung Chung, the lead on the paper.

### Corrected

With the feedback in mind, I decided to pursue a more faithful (less lazy) implementation.

The change for `MultiRNNCell` is actually minimal:

```python
class GFMultiRNNCell(rnn_cell.MultiRNNCell):
    """
    MultiRNNCell composed of stacked cells that interact across layers
    Based on http://arxiv.org/pdf/1502.02367v4.pdf
    """

    def __init__(self, cells):
        for cell in cells:
            if not isinstance(cell, GFCell):
                raise ValueError("Cells must be of type GFCell")

        super(FeedbackCell, self).__init__(cells)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Conveniently the concatenation of all hidden states at t-1
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("Cell%d" % i):
                    cur_state = array_ops.slice(
                            state, [0, cur_state_pos], [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                    cur_inp, new_state = cell(cur_inp, cur_state, state)
                    new_states.append(new_state)

        return cur_inp, array_ops.concat(1, new_states)
```

The operative changes is that that every individual cell now has access to the `state` var, which is $h_{t - 1}^*$.

```python
cur_inp, new_state = cell(cur_inp, cur_state, state)
```

There's also a type check to make sure you're composing the cell stacks from cells with the gated feedback capability.

We then define a `GFCell` class that looks very much like the `RNNCell` class with a few enhancements:

```python
class GFCell(object):
    """Abstract object representing a cell in a Gated Feedback RNN
    Operates like an RNNCell
    """

    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        """Return state tensor (shape [batch_size x state_size]) filled with 0.

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          A 2D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        zeros = array_ops.zeros(
                array_ops.pack([batch_size, self.state_size]), dtype=dtype)
        zeros.set_shape([None, self.state_size])
        return zeros

    def __call__(self, inputs, state, full_state, layer_sizes, scope=None):
        raise NotImplementedError("Abstract method")

    def compute_feedback(self, inputs, full_state, layer_sizes, scope=None):
        with tf.variable_scope("Global Reset"):
            cur_state_pos = 0
            full_state_size = sum(layer_sizes)
            summation_term = tf.get_variable("summation", self.state_size, initializer=tf.constant_initializer())
            for i, layer_size in enumerate(layer_sizes):
                with tf.variable_scope("Cell%d" % i):
                    # Compute global reset gate
                    w_g = tf.get_variable("w_g", self.input_size, initializer=tf.random_uniform_initializer(-0.1, 0.1))
                    u_g = tf.get_variable("u_g", full_state_size, initializer=tf.random_uniform_initializer(-0.1, 0.1))
                    g__i_j = tf.sigmoid(tf.matmul(inputs, w_g) + tf.matmul(full_state, u_g))

                    # Accumulate sum
                    h_t_1 = \
                        tf.slice(
                                full_state,
                                [0, cur_state_pos],
                                [-1, layer_size]
                        )
                    cur_state_pos += layer_size
                    U = tf.get_variable("U", [self.input_size, self._num_units],
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1))
                    b = tf.get_variable("b", self.state_size, initializer=tf.constant_initializer(1.))
                    summation_term = tf.add(summation_term, g__i_j * tf.matmul(U, h_t_1) + b)

        return summation_term
```

A few other pieces are needed in `__call__`, such as the entire previous hidden state and the layer sizes. We then implement the summation described in the paper.
The modified hand term in the paper (e.g. used to replace Eq 9) is always a summation term. For each previous timestep hidden layer $i$, we compute $g^{i \to j}$ and this summation $S^j$.


### Tensorflow Canonical

Due to time constraints and the fact that more abstract/general architectures have come out since I started this blog post, I won't actually be doing a TF-canonical implementation to contribute.