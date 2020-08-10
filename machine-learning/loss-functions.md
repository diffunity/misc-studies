# Loss Functions
> import libraries
<pre><code>import torch
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras.metric import *
import tensorfflow.keras.backend as B
</code></pre>

## Pytorch

> variable setting
<pre><code>target = torch.tensor([1,2,3])
inp = torch.full((target.shape[0],10), 0.5)
target, inp
</code></pre>
<pre>(tensor([1, 2, 3]), 
tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
        [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000], 
        [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]))
</pre>

* #### Negative Log Likelihood loss
$$ l_n = -w_{y_n}x_{n,y_n}, w_c = weight[c] * 1 )\{c \neq \textnormal{ignore\_index}\}$$
weight defaults to 1s
<pre><code>F.nll_loss(inp, target)
</code></pre>

<pre>tensor(-0.5000)
</pre>

* #### Cross Entropy loss
<pre><code>F.cross_entropy(inp, target)
</code></pre>

<pre>tensor(2.3026)
</pre>

* #### KL-divergence loss
allows you to compute the difference between two distributions (hence both inputs must be distributions)

> variable setting for kl divergence
<pre><code>one_hot = torch.zeros_like(inp)
one_hot.scatter_(1, target.unsqueeze(-1),1)
</code></pre>

<pre>tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])</pre>

> torch implementation $F.kl\_div(y, \hat{y})$:
> $$ \hat{y} \cdot ( log(\hat{y} - y )$$
> tf implementation $KLDivergence(\hat{y},y)$: 
> $$ \hat{y} \cdot log(\frac{\hat{y}}{y}) $$


<pre><code>tf.keras.losses.KLDivergence()(one_hot, inp), \
F.kl_div(torch.log(inp), one_hot, reduction="batchmean")
</code></pre>

<pre>(tf.Tensor: shape=(), dtype=float32, numpy=0.6931332, tensor(0.6931))
</pre>

> reduction ="batchmean" parameter ensures kl_div is averaged along the batch only and not through the whole dimension of the input (here its 10) (this bug will be updated as default in the next version of pytorch)

#### Comparisons

* #### NLL and CrossEntropy
both takes target as the position values before one-hot encoding
<pre><code>F.nll_loss(torch.log_softmax(inp, 1), target), \
F.cross_entropy(inp, target)
</code></pre>

<pre>(tensor(2.3026), tensor(2.3026))
</pre>

* #### NLL and KL-Divergence
KL-Divergence takes 2 
the advantage of using KL-divergence is it allows you to modify the label strength (and not just use a simple one hot encoding as default like it is in NLL or cross entropy)
<pre><code>F.nll_loss(inp, target), \
F.kl_div(inp, one_hot, reduction="batchmean")
</code></pre>

<pre>(tensor(-0.5000), tensor(-0.5000))
</pre>

* #### CrossEntropy and KL-Divergence
<pre><code>F.cross_entropy(inp, target), \
F.kl_div(torch.log_softmax(inp, 1), one_hot, reduction="batchmean")
</code></pre>

<pre>(tensor(2.3026), tensor(2.3026)))
</pre>

* #### NLL & CrossEntropy & KL-Divergence
<pre><code>F.nll_loss(torch.log_softmax(inp,1), target), \
F.cross_entropy(inp,target), \
F.kl_div(torch.log_softmax(inp, 1), one_hot, reduction="batchmean")
</code></pre>

<pre>(tensor(2.3026), tensor(2.3026), tensor(2.3026))
</pre>

## Tensorflow

> variable setting

``` python
target = tf.convert_to_tensor([1,2,3])
inp = tf.fill((target.shape[0],10),0.5)
target, inp
```

```
(<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>,
 <tf.Tensor: shape=(3, 10), dtype=float32, numpy=
 array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=float32)>)
```

* #### Categorical Cross Entropy

> variable setting for CategoricalCrossEntropy
<pre><code>one_hot = tf.one_hot(target, inp.shape[1], axis=-1)
</code></pre>

```
<tf.Tensor: shape=(3, 10), dtype=float32, numpy=
array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]], dtype=float32)>
```

<pre><code>loss = tf.keras.losses.CategoricalCrossentropy()(one_hot, inp)
</code></pre>

```
<tf.Tensor: shape=(), dtype=float32, numpy=2.3025851>
```