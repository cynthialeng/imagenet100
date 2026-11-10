# HW3 Local global feature fusion

**Task and data.** Classify tabby cats, cauliflower, and computer keyboards
from ImageNet100. Reuse HW2's 600 training, 150 validation, and 150 test images
in `data/hw2/`. Images are RGB, resized to 32×32, and scaled to 0–1.

**Baseline.** MLP: 3,072 → 128 → 64 → 3, with ReLU after the hidden layers.

**Modification.** Add a CNN branch with two 3×3 convolutions (16 and 32 channels),
ReLU and 2×2 max pooling after each, global average or max pooling, and a
32→64 projection with ReLU. Combine its 64 features L with the MLP's 64
features G using `F = αL + (1−α)G`, then classify with a 64→3 layer.
Fixed fusion uses α=0.5; learned fusion uses `sigmoid(Linear([L,G]))`,
initialized at 0.5. The hypothesis was that adaptive local/global fusion would
improve classification. Ablations: MLP only, CNN only, fixed and learned fusion.

## Run

From `imagenet100-viewer/`, activate your Python environment and install:

```bash
python -m pip install -r backend/requirements.txt
```

Evaluate the saved checkpoints:

```bash
python -m training.hw3.evaluate_test
```

Reproduce all experiments in new folders:

```bash
python -m training.hw3.train
```

Defaults: 100 epochs, seed 42, batch size 32, cross-entropy loss, SGD with learning rate 0.01,
no momentum or weight decay. Shared components have identical
initial weights and all variants use the same batch order. Checkpoints chosen
by lowest validation loss; test data does not guide training or selection.
Each run retains `config.json`, weights, history, summary, and fusion gate logs.
Final results are in `outputs/analysis/`.

## Results

| Model | Global pooling | Test loss | Test accuracy |
|---|---|---:|---:|
| MLP | None | 0.8554 | 63.3% |
| CNN | Average | 1.0742 | 52.7% |
| CNN | Max | 1.0406 | 57.3% |
| Fixed fusion | Average | 0.8333 | 66.0% |
| Fixed fusion | Max | 0.8330 | 66.0% |
| Learned fusion | Average | 0.8318 | 64.7% |
| Learned fusion | Max | 0.8308 | 64.7% |

Fixed fusion made four more correct predictions than the MLP on 150 images.
The learned gate favored the MLP (mean CNN weight about 0.10–0.11) and did not
consistently outperform fixed fusion. Max pooling helped the CNN but barely
changed fusion. Both CNNs were still improving at epoch 100.
Fusion took about 0.43–0.45 seconds per epoch versus 0.26 for the MLP.
These are small, single-seed results; the test split was already used in HW2.
