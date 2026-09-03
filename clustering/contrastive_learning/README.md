# Contrastive Learning Clustering Engine

This engine utilizes Contrastive Learning, a self-supervised approach, to learn robust representations of optical network states. By pulling augmented versions of the same state together and pushing distinct states apart, the engine organizes the state space into distinct clusters corresponding to different types of soft failures or normal operations.

## Mathematical Details

### InfoNCE Loss

The core of the contrastive learning clustering engine relies on the Noise Contrastive Estimation (InfoNCE) loss. Given a batch of $N$ states, each state is augmented to create two correlated views, resulting in $2N$ data points.

For a positive pair $(z_i, z_j)$ generated from the same underlying state $x$, the contrastive loss function for $z_i$ is defined as:

$$ \ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)} $$

where:
- $z = g(f(x))$ is the projection of the representation.
- $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ is the cosine similarity.
- $\tau$ is a temperature parameter that scales the distribution of similarities.
- $\mathbb{1}_{[k \neq i]}$ is an indicator function evaluating to 1 if $k \neq i$ and 0 otherwise.

### Total Loss

The total loss is computed across all positive pairs in the batch:

$$ \mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} [\ell_{2k-1, 2k} + \ell_{2k, 2k-1}] $$

## Technical Details

- **Data Augmentation**: Requires domain-specific augmentations for optical networks (e.g., adding noise to OSNR values, masking node features, perturbing link capacities).
- **Encoder Architecture**: Employs deep architectures like Graph Neural Networks (GNNs) or Transformers to extract feature representations $f(x)$.
- **Projection Head**: A small MLP $g(\cdot)$ maps the representations to the contrastive space.
- **Integration with RL**: The learned representations provide a rich, structured observation space for the RL agents, facilitating faster convergence and better generalization in failure detection and localization tasks.
