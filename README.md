# Multilayer Perceptron

A from-scratch (NumPy) MLP that classifies breast-cancer diagnoses (Malignant / Benign).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## How to run

Run the scripts from inside `src/` in this order:

```bash
cd src
```

### 1. (optional) Explore the data

```bash
python explore.py --data ../data.csv
```

### 2. Prepare / split the data

Reads the raw CSV, standardizes features, and writes `data_train.npz` + `data_val.npz`.

```bash
python prepare_data.py --data ../data.csv --train_ratio 0.8
```

### 3. Train the model

Trains the MLP and saves it to `mlp_model.npz`.

```bash
python train.py --epochs 100 --batch_size 32 --lr 0.01 --hidden 5 10
```

### 4. Predict / evaluate

Loads the saved model and reports accuracy on the validation set.

```bash
python predict.py --data data_val.npz --model mlp_model.npz
```

## Scripts

| Script            | Purpose                                     | Key flags |
|-------------------|---------------------------------------------|-----------|
| `explore.py`      | EDA + plots on the raw CSV                   | `--data`, `--test_size` |
| `prepare_data.py` | Standardize + train/val split → `.npz`      | `--data`, `--train_ratio`, `--train_out`, `--val_out` |
| `train.py`        | Train MLP, plot metrics, save model         | `--epochs`, `--batch_size`, `--lr`, `--hidden`, `--train`, `--val`, `--model` |
| `predict.py`      | Load model, report sigmoid/softmax accuracy | `--data`, `--model` |
| `mlp.py`          | `MLP` class (library, not run directly)     | — |
| `io_utils.py`     | Shared data/model load & save helpers       | — |

Every script supports `-h` / `--help` for the full flag list.

## Concepts

- **Feedforward** — inputs propagate layer by layer through weighted sums + sigmoid activations.
- **Backpropagation** — errors propagate backward to compute per-layer gradients.
- **Gradient descent** — weights/biases updated by the gradient scaled by the learning rate.

## Regex (label column extraction)

```
(?<=^[^,]*,[^,]*,[^,]*,[^,]*,[^,]*),.*
```
