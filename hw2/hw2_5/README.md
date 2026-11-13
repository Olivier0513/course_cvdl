# ResNet50
## Residual Connection
### Degradation Problem
When neural network goes deeper, the training error may actually increase, this problem calls `Degradation Problem`.

`Degradation Problem` will cause Training Error increase.
### Residual Learning
While training:
- Input: $x$
- Output: $H(x)$

ResNet add `Residual`:
- Residual, $F(x)=H(x)-x$
- $H(x)=F(x)+x$

When $F(x)=0$, $H(x)=x$
- Assume the newly added layers provide no improvement, preserve the original input.
### Residual Block
- Every `Conv Layer` is follow by a `BatchNorm Layer`.
- Between each `Conv Layer` add a `ReLU Layer`.
- Each `Residual Block` has two or more `Conv Layer`, and has one `Shortcut`.
## Bottleneck
![alt text](https://user-images.githubusercontent.com/36760800/114392408-05878080-9b99-11eb-9999-f37b87cdc348.png)
### Basic Block
- 2 `Conv Layer` with kernel size 3x3 and 64 channels.
- For ResNet-18 and ResNet-34.
### Bottleneck Block
Bottleneck block structure:
- 1x1 `Conv Layer` with 64 channels.
- 3x3 `Conv Layer` with 64 channels.
- 1x1 `Conv Layer` with 256 channels.

Bottleneck reduces the number of channels during the 3×3 convolution, thereby reducing the number of parameters.
## Fully Connected Layer
Global Average Pooling + FC Layer
# hw2_5
## Main Issue: Imbalanced Dataset
The dataset contains more dog images than cat images.
### Loss1: Binary Cross Entropy（BCE）
$$L_{BCE}=-[y\log(p)+(1-y)\log(1-p)]$$
- $y$ is label, 0 or 1
- $p$ is predict outcome
- BCE Loss is small when:
    - $y=1$ and $p$ is close to $1$ $(-y\log(p))$
    - $y=0$ and $p$ is close to $0$ $(-(1-y)\log(1-p))$
### Loss2: Focal Loss
$$L_{FL}=-\alpha_t(1-p_t)^\gamma\log(p_t)$$
- When a sample is classified correctly with high confidence (i.e., $p_t \to 1$), $(1 - p_t)^\gamma$ approaches 0, significantly reducing the contribution of this "easy sample" to the overall loss.
- γ controls the strength of the down-weighting. When γ = 0, Focal Loss reduces to standard Cross-Entropy
- $\alpha_t$ adjusts the weights of different classes, increasing the contribution of minority classes to the loss