# Multilayer Perceptron

From-scratch (NumPy) MLP classifying breast-cancer diagnoses (Malignant / Benign)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

## How to run

### 1. Split the dataset

```bash
python src/prepare.py --data data.csv --train_ratio 0.8 --seed 42
```

Reads the raw CSV, standardizes features, writes `data_train.npz` + `data_val.npz`.
`seed` makes the split repeatable (same shuffle every run).

### 2. Train

```bash
python src/train.py \
  --epochs 100 --batch_size 32 --lr 0.01 --hidden 5 10 7 \
  --loss categoricalCrossentropy --activation sigmoid \
  --optimizer adam --patience 10
```

Trains the MLP (≥2 hidden layers, softmax output), prints train/val loss every
epoch, shows loss + accuracy curves at the end, and saves the model
(topology + weights) to `mlp_model.npz` and metrics history to `mlp_history.json`.

### 3. Predict / evaluate

```bash
python src/predict.py --data data_val.npz --model mlp_model.npz
```

Loads the saved model, predicts on the given set, reports accuracy, binary
cross-entropy, precision, recall, and F1.

### 4. Compare multiple runs 

```bash
python src/compare.py --run_log tag.csv --tags small big
```

Reads tags from the run log (default: `tag.csv`, all tags if `--tags` is
omitted), loads each run's saved history, and plots validation-loss curves
together. Missing history files are skipped with a warning.

## Concepts

Ordered as they occur in a training run: unit → parameters → hyperparameters →
forward pass → loss → backward pass → weight update → stopping → evaluation.

### Perceptron

The base unit of the network: one neuron with one or more inputs, an activation
function, and a single output. Two steps produce its output:

1. **Weighted sum** — `z = Σ(xₖ · wₖ) + bias`, over all `N` inputs of the previous
   layer.
2. **Activation** — `a = f(z)`, squashing `z` into the neuron's output range.

### Weights, bias, and initialization

- **Weight** — a learned parameter scaling one input's contribution to the
  weighted sum.
- **Bias** — a learned constant added to the weighted sum; shifts the
  activation threshold independently of the inputs (implemented as an always-on
  "neuron" with output 1).
- **He initialization** (`initialize_weights`, `mlp.py`) — weights are drawn as
  `W ~ N(0, 2/fan_in)`, the default here since it pairs well with sigmoid/ReLU
  hidden layers and keeps gradients from vanishing/exploding at the first
  forward pass.

### Hyperparameters

Values set before training starts (not learned):

- **Hidden layer** — a layer of neurons between input and output, not directly
  observed (`--hidden 5 10` = two hidden layers of size 5 and
  10). The subject requires at least two by default.
- **Epoch** — one complete pass of the training set through the model:
  every sample is fed forward, the loss is computed, and weights are updated
  via backpropagation.
- **Batch size** — number of examples processed together
  before one gradient-descent weight update. Smaller batches update more often
  per epoch but with noisier gradients; larger batches give smoother but
  slower updates.
- **Learning rate** — scale factor applied to the gradient before
  updating each weight (`w -= lr · ∂loss/∂w`). Too high and the loss diverges;
  too low and training crawls.

### Feedforward

The forward pass: data flows input → hidden layer(s) → output, one layer at a
time, each computing `a⁽ˡ⁾ = f(a⁽ˡ⁻¹⁾ · W⁽ˡ⁾ + b⁽ˡ⁾)`. No information flows
backward during this step — hence "feedforward". `f` is the **activation
function** (`--activation` for hidden layers; the output layer is always
softmax):

- **Sigmoid** — `σ(z) = 1 / (1 + e⁻ᶻ)`. Used on hidden layers. Squashes any
  input to `(0, 1)`; its derivative `σ'(z) = σ(z) · (1 − σ(z))` is what
  backpropagation uses to push the error back through the layer.
- **Softmax** — `softmax(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ`. Used on the output layer only.
  Turns the two raw output scores into a probability distribution over
  Malignant/Benign that sums to 1.

### Backpropagation

The **gradient** (`∂loss/∂W`) is how much each weight contributed to the loss.
Backpropagation computes it for every weight by propagating the loss backward
through the network, layer by layer, using the **chain rule** — without
recomputing the whole forward pass per weight. For a softmax output paired with
cross-entropy loss, the two gradients simplify into a single term:
`∂loss/∂z_output = output − y`. That error is then pushed back through each
hidden layer as `δ⁽ˡ⁻¹⁾ = (δ⁽ˡ⁾ · W⁽ˡ⁾ᵀ) · σ'(a⁽ˡ⁻¹⁾)`.

### Gradient descent

Updates each weight/bias in the direction that reduces the loss, scaled by the
**learning rate** (`--lr`): `w -= lr · ∂loss/∂w`. Training here uses
**mini-batch gradient descent** (`--batch_size`): gradients are averaged over a
small batch of examples rather than one example (noisy, slow) or the whole
dataset (stable, slow) at a time.

### Optimizers

Alternatives to plain gradient descent that use a running average of past
gradients to take smarter steps, selected with `--optimizer`:

- **Momentum** — accumulates a velocity `v = β·v + (1-β)·grad`, then steps by
  `v`; smooths out oscillations.
- **RMSprop** — divides each step by a running average of squared gradients,
  so parameters with noisy/large gradients get smaller steps.
- **Adam** — combines momentum (first moment) and RMSprop (second moment),
  with bias-correction terms (`/ (1 - βᵗ)`) so early steps aren't
  underestimated.

### Loss function — categorical cross-entropy

Measures how far the predicted probability distribution is from the true
one-hot label; the number minimized during training.

`loss = -mean(Σ y · log(p))` over the two classes, per example.

`predict.py` additionally reports the equivalent **binary cross-entropy**
form the subject asks for:
`E = -1/N · Σ [yₙ·log(pₙ) + (1-yₙ)·log(1-pₙ)]`, where `p` is the predicted
probability of the Malignant class.


### Early stopping

Halts training once validation loss stops improving for `--patience`
consecutive epochs (`0` = disabled), keeping the model from overfitting to the
training set after it has stopped generalizing.

### Evaluation metrics

`predict.py` reports, from the confusion counts (TP/TN/FP/FN) of predicted vs.
actual labels:

- **Accuracy** — `(TP + TN) / total`.
- **Precision** — `TP / (TP + FP)`: of predicted-Malignant, how many really are.
- **Recall** — `TP / (TP + FN)`: of actually-Malignant, how many were caught.
- **F1** — `2 · precision · recall / (precision + recall)`: harmonic mean of
  the two, useful when class counts are imbalanced.

## Training loop (one epoch)

```
Input → Feedforward (Input → Hidden → Output) → Prediction
      → Loss calculation → Backpropagation (Output → Hidden → Input)
      → Gradient descent update → repeat next batch
```
