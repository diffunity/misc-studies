## Graph Neural Networks

[Microsoft Lecture Part 1](https://www.youtube.com/watch?v=zCEYiCxrL_0)

[Microsoft Lecture Part 1](https://www.youtube.com/watch?v=cWIeTMklzNg)

[Graph Convolutional Network](https://tkipf.github.io/graph-convolutional-networks/)

Graph Neural Networks are semi-supervised learning method that applies neural networks to learn a graph representation of a real-world problem
### GNN Workflow

1. Graph representation of a real-world problem
2. GNN modelling 

### Graph Structure
* Graphs \
Data is modelled as graphs
$$G = (V, E)$$

* Vertices \
Each node contains a vector representation of an object 

* Edges \
Each edge contains the relationship between its vertices

* Modelling representations
    - Each feature $x_i$ for every node $i$ is summarized in a $N \times D$ feature matrix $D$ ($N$ : number of nodes, $D$ : number of input features)
    - The graph sturcture is typically represented in an $N \times N$ adjacency matrix $A$, where each value represents the edges 


### General Learning Process
Define learning time steps (eg. $s$)

* Node updating function
$${h}_{t}^{n} = q {(}{{h}^{n}_{t-1}}, \underset{\forall n_{j}:n \stackrel{k}\rightarrow n_j}{\bigcup}f_{t}(\bold{h}^{n}_{t-1},\bold{h}^{n_j}_{t-1}) {)} $$

* Message function
$$\underset{\forall n_{j}:n \stackrel{k}\rightarrow n_j}{\bigcup}f_{t}(\bold{h}^{n}_{t-1},\bold{h}^{n_j}_{t-1})$$

eg. dimensionality reduction

The idea of updating is called "passing on the message"

* Unsupervised Process

Imagine a clock,

For every tick, each node updates its value from the neighbors

There is no notion of convergence. How many times you wanna update each node is hyperparameter. (a.k.a. n-order neighborhood information)

With an adequate number of "message passing", even a randomly initialized weights can properly learn the relationship between the nodes, and a latent representation of the graph

* Supervised Process

The latent representation produced at the end of the unsupervised process can be used for downstream tasks such as

1. node selection : utilize the learned vector

2. node classification : with a given label, conduct supervised learning

3. graph classification : summarize all the vectors in the graph

4. etc.

With these tasks, we can optimize the weights for our specific tasks

#### Modelling problems

1. a way to prepare the messages (f function)

2. a way to summarize the received information (Union logic)

3. a way to update the state (q function)


### GCN for text classification

* Difference with GCN
    * Pooling Function 
$$ L^{(j+1)} = p(\hat{A}L^{(j)}W_{j})$$

* Representation
  * $(V + D) \times (V + D)$ Adjacency matrix where $V$ = number of vocabularies and $D$ = number of documents
  * $\textnormal{PMI}(i,j)$ where   $i,j \textnormal{ are words, PMI}(i,j)>0$ 
  * $\textnormal{TF-IDF}_{i,j}$ where $i \textnormal{ is document, }j\textnormal{ is word}$
  * $1$ where $i = j$
  * $0 \textnormal{ otherwise}$

> Two layers of GCN with ReLU : second layer node (word + document) embeddings have the same size as the labels set + softmax
