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

### Feedforward
A process of passing input data through a neural network to generate a final prediction.
```
Input (X)
      │
      ▼
Weight × Input + Bias
      │
      ▼
 Sigmoid
      │
      ▼
Hidden Layer 1
      │
      ▼
Weight × Input + Bias
      │
      ▼
 Sigmoid
      │
      ▼
Hidden Layer 2
      │
      ▼
Weight × Input + Bias
      │
      ▼
 Sigmoid
      │
      ▼
Output
```
**Weight**

**Bias**

**Activation Function**
- Introduce **non-linearity** into the neural network.
- Enable the model to learn **complex** patterns and relationships.
- Transform the output of each neuron before passing it to the next layer.
- Improve the network's ability to solve classification and regression problems.
- Support backpropagation because most activation functions are differentiable.

### Backpropagation
#### Calculate loss
errors propagate backward to compute per-layer gradients.

### Gradient descent
weights/biases updated by the gradient scaled by the learning rate.

### softmax

### one epoch
```
① Input
      │
      ▼
② Feedforward
(Input → Hidden → Output)
      │
      ▼
③ Prediction
      │
      ▼
④ Loss Calculation
      │
      ▼
⑤ Backpropagation
(Output → Hidden → Input)
      │
      ▼
⑥ Update Weights
      │
      ▼
repeat next data
```
## Regex (label column extraction)

```
(?<=^[^,]*,[^,]*,[^,]*,[^,]*,[^,]*),.*
```
