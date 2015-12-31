---
layout: post
title: 'Comparing Highway Networks & GradNets'
date: 2015-12-15 17:12
comments: true
categories:
---

## tl;dr

Highway Networks employ more learnable weights for a similar network, but are easier to use than GradNets. Architectural interpolation for very deep nets seems to work better when machines do the work.

## Intro

As part of a deep learning study group, I implemented both Highway Networks and GradNets to compare results under similar conditions. The Highway Network implementation is based on Jim Fleming's [blog post](https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.7z1d6allb) with small  modifications, and the GradNet implementation is derivative of that. Both demos are in Jupyter Notebooks, run Tensorflow, and do not require GPUs to finish quickly. To keep formulations simple, I only compare fully-connected networks.

## Highway Networks

Highway Networks are an architectural feature that allows the network to adaptively "flatten" itself by passing certain neurons through without any transformation. The network typically starts with initial biases towards passing the data through, behaving like a shallow neural network. After some training, the weights that control the "gates" of the network start to adjust and close down the highway in the early layers. Certain "lanes" of the highway will selectively activate.

These networks in fact learn not just the weights for the underlying affine transformations that are then run through the nonlinearn activation kernels (sigmoid, ReLU, tanh, etc), but also a companion set of weights for the gate that determines how much of that activation to use. This gate is controlled by a sigmoid activation applied to an affine transform, parameterized by the companion weights.

The paper describes these more formally as the Hypothesis $H$, the Transform Gate $T$, and the Carry Gate $C$. The value of the Carry Gate is simply 1 minus the value of the Transform Gate. The Hypothesis is the underlying transformation being performed at the layer.

A standard fully connected layer looks like

$$
y = H(x, W_H) = activation(W_H^Tx + b_H)
$$

To this:

$$
y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C)
$$

$\cdot$ denotes elementwise multiplication. Note that all W matrices match in dimension.

Since $T(x, W_T) = 1 - C(x, W_C)$, the last term in the equation above can becomes $1 - T(x, W_T)$

$T(x, W_T) = sigmoid(W_H^Tx + b_T)$ produces element-wise "gates" between 0 and 1.

A critical point of initialization is $b_T$. It should be set to a fairly negative value so that the network initially passes the $x$ through.

In code:

```python
def highway_layer(x, size, activation, carry_bias=-1.0):
    W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[size]), name='bias')

    W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name='weight_transform')
    b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name='bias_transform')

    H = activation(tf.matmul(x, W) + b, name='activation')
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
    C = tf.sub(1.0, T, name="carry_gate")

    y = tf.add(tf.mul(H, T), tf.mul(x, C), 'y')
    return y
```

More about highway networks on this [page](http://people.idsia.ch/~rupesh/very_deep_learning/) and the papers listed there.


## GradNets

[GradNets](http://arxiv.org/abs/1511.06827) offer a simplified alternative to gradual interpolation between model architectures. The inspiration is similar to that of Highway Networks; early in training, prefer simpler architecture, whereas later in training, transition to complex.

The variable $g$ anneals over a $\tau$ epochs, controlling the amount of interpolation between the simple activation and the nonlinear one. Using similar notation as before:

$$
g = \min(t / \tau, 1)
$$

$$
H(x, W) = ReLU(W^Tx + b)
$$

$$
J(x, W) = I(W^Tx + b)
$$

$$
y = g \cdot H(x, W) + (1 - g) \cdot J(x, W)
$$

In code:

```python
def grelu_layer(x, input_size, output_size, g):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='bias')
    u = tf.matmul(x, W) + b
    y = g * tf.nn.relu(u) + (1 - g) * u
    return y
...
for i in range(num_iter):
  batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)

  epoch = i / iter_per_epoch
  gs = min(epoch / tau, 1.0)
...
```

I used `__future__.division` to default to floating point division with a single `/`, whereas integer floor division would be `//`. The paper was not explicit on which one to use, but it made sense to be as gradual as possible in GradNets.


## Experiment

### Highway Network

I was able to reproduce results on Highway Networks quite easily. Using the following parameters:

* 50 hidden layers of size 50 each
* Minibatch size of 50
* SGD optimizer
* $10^{-2}$ starting learning rate
* No weight decay
* Initial carry bias of -1

I got to ~92% test accuracy within 2 epochs and hit the best test accuracy of ~96% around epoch 13. The network started to overfit after that, which is expected because I did not apply learning rate decay or any form of regularization.

I tried a few other configuration and referred to Jim's post (linked above) in order to confirm that the model converged under a variety of conditions.

### GradNets - Linear GReLU

I tried to reproduce the first example from GradNets, interpolating between a simple linear (Identity) activation and a ReLU activation. The underlying weights are still the same under each path, so the output is weighted mix of 2 different activations on the same affine transformation.

Using the same optimizer (SGD), the same learning rate, and otherwise the same architecture as Highway Networks, I was not able to get the network to converge. The norm of the gradients moved toward 0 as `g` annealed to 1, and when g hit 1, the gradients all hit 0.

### GradNets v2 - Identity GReLU

As an alternative approach, I tried to obtain an interpolation closer to what Highway Networks achieved. Following the same constraint as Highway Networks, I modified the GReLU layer to interpolate between the full transformation and an identity function directly on $x$ rather than on the affine transform $W^Tx + b$. Here is the revised output:

$$
g \cdot ReLU(W^Tx + b) + (1 - g) \cdot I(x)
$$

The GReLU layer now looks like:

```python
def grelu_layer(x, input_size, output_size, g):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='bias')
    y = g * tf.nn.relu(tf.matmul(x, W) + b) + (1 - g) * x
    return y
```

Because of the multiplicative nature of backpropogation, I ended up with situations where I had exploding gradients and weights. I added a bit of monitoring and noted that the gradients tended to explode as `g` crossed $0.4 - 0.7$. 

To understand how this happens, here's an example. Suppose `y_` is non-zero and `y` is very close to zero. `log(y)` is a very big negative number, and gets multiplied by a non-zero `- y_`. When gradients flow more freely as `g` increases, this large number is multipled and summed across many nodes in many layers.

The value of gradient $dW$ at a given layer is $dx \times dy$. $dy$ is that big number that came from the next layer, and $dx$ is the activation from previous layer. This can lead to exponential growth in the gradient under the wrong conditions.

A good proxy for big activations is big weights, so I collected those as well. I noticed some weight explosion, so I applied the `relu6` activation, which clips the output of the unit. It prevented gradient explosion, but at saturation in hidden layers, the network quickly diverges without means to recover.

### Kitchen Sink Fix

Without making the network any shallower, I tried:

* Regularization: dropout, L1, L2
* Optimizer: Adam, AdaGrad, RMSProp
* Learning Rate: start rate, exponential and constant decay
* Gradient clipping and normalization

Nothing seemed to help the 50-layer Linear GReLU train. As for the Identity GReLU (the one that more resembles Highway Networks), it took a combination of:

* L1 regularization - very carefully chosen to stabilize weights
* Lower starting learning rate of $10^{-3}$
* Aggressive exponential decay of the learning rate
* Using the Adam optimizer
* Mild dropout (5%)

...in order to keep the Identity GReLU from diverging, but it would still hit a random `NaN` spike that didn't seem to follow  a climb in the L1 norm of weights & gradients. I could have considered checkpointing + early-stopping, but I would prefer that the network demonstrate stability without outside help. It certainly would've saved a lot of time.

Eventually, I figured out from reading forum posts that the Identity GReLU spikes when $y \to 0$ and $log(y) \to -\infty$. Doh! Lesson learned: read the forums!

The simple fix is to add a small number to `y`:

```python
cross_entropy = -tf.reduce_sum(y_ * tf.log(y + 1e-9))
```

I ended up hitting 94% accuracy around epoch 15 and staying there until the end of training. I was pretty happy with the graphs for weights and gradients - they remained in a pretty small range throughout. It should be emphasized that the convergence property was sensitive to ALL of the hyperparameters above. Significant changes in any of them led to divergence or no learning.

### Sanity Check

For my own sanity check, I left the hyperparameters in their tuned state and removed the GradNet portion to see how well the network would do. As expected, the network failed to bounce out of its initial state.

## Conclusion

While Highway Networks effectively double the number of parameters per layer, they work surprisingly well with naive hyperparameters.

In contrast, GradNets are highly sensitive to hyperparameters. Painstaking, manual search (which could have been automated through Grid- or Randomized-Search, but still require some knowledge and tweaking) and weight/gradient debugging were required to deliver usable results. Applying a heuristic, namely a $g$-weighting that annealed over a number of epochs, worked best when interpolating between "Highway Mode" (Identity) and ReLU. Interpolating the nonlinearity (Linear) didn't seem to help with a 50-layer network.

The winning characteristic of Highway Networks is ability to learn good gating parameters for every neuron as part of end-to-end training. The Identity GReLU does a fair job of approximating this without the additional model parameters, but
(1) it is only able to so with one $g$ across the entire network, and
(2) only succeeds under very narrow conditions that must be discovered through hyperparameter search.

To the credit of GradNets, gradual interpolation of other aspects of the architecture, such as dropout, batch normalization, convolutions, are available and have no counterpart in Highway Networks. Further investigation into using Highway Networks for these components could be interesting.

Demo runs with inline implementation can be found [here](https://github.com/ZhangBanger/highway-vs-gradnet)
