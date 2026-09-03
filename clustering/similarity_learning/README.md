# Similarity Learning Clustering Engine

This engine utilizes metric learning techniques to learn a similarity function between different network states. The primary goal is to cluster optical network states by grouping those that represent similar soft failure manifestations, enabling more effective downstream classification and detection.

## Mathematical Details

### Objective Function

Given a set of network state representations $\mathcal{X} = \{x_1, x_2, \ldots, x_N\}$, the similarity learning model learns a mapping function $f_\theta: \mathcal{X} \rightarrow \mathbb{R}^d$ parameterized by $\theta$.

The objective is to minimize a contrastive-based loss or triplet loss. For triplet loss, given an anchor $x_a$, a positive sample $x_p$ (similar state), and a negative sample $x_n$ (dissimilar state), the loss is defined as:

$$ \mathcal{L}(\theta) = \sum_{i=1}^{N} \max(0, \|f_\theta(x_a^{(i)}) - f_\theta(x_p^{(i)})\|_2^2 - \|f_\theta(x_a^{(i)}) - f_\theta(x_n^{(i)})\|_2^2 + \alpha) $$

where $\alpha$ is a margin enforcing a minimum distance between positive and negative pairs.

### Clustering Strategy

Once the similarity space is learned, standard clustering algorithms like K-Means or DBSCAN can be applied on the learned embeddings $f_\theta(x)$ to find discrete clusters representing soft failure types.

## Technical Details

- **Input Modality**: The engine processes node features, edge features, and spectral utilization metrics.
- **Model Architecture**: Typically a Multi-Layer Perceptron (MLP) or a Graph Neural Network (GNN) layer is used as the base encoder $f_\theta$.
- **Integration**: The learned representations serve to partition the state space for the hierarchical reinforcement learning agents, simplifying their policy learning process.
