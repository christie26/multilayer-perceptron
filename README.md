# Multilayer Perceptron

A from-scratch (NumPy) MLP that classifies breast-cancer diagnoses (Malignant / Benign).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

## How to run

### 1. Explore the data

```bash
python src/explore.py
```

### 2. Prepare / split the data

Reads the raw CSV, standardizes features, and writes `data_train.npz` + `data_val.npz`.

```bash
python src/prepare.py
```

### 3. Train the model

Trains the MLP and saves it to `mlp_model.npz`.

```bash
python src/train.py --epochs 100 --batch_size 32 --lr 0.01 --hidden 5 10
```

### 4. Predict / evaluate

Loads the saved model and reports accuracy on the validation set.

```bash
python predict.py --data data_val.npz --model mlp_model.npz
```

## Concepts

- **Feedforward** — inputs propagate layer by layer through weighted sums + sigmoid activations.
- **Backpropagation** — errors propagate backward to compute per-layer gradients.
- **Gradient descent** — weights/biases updated by the gradient scaled by the learning rate.

## Regex (label column extraction)

```
(?<=^[^,]*,[^,]*,[^,]*,[^,]*,[^,]*),.*
```
