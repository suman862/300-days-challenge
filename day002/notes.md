

###  Why Initialize Weights Carefully?

Neural networks are **sensitive to how their weights are initialized**. Poor initialization can lead to:

* **Vanishing/exploding gradients** (especially in deep networks)
* **Slow convergence**
* **Getting stuck in poor local minima**

This is why techniques like **Xavier (Glorot)** and **He initialization** are used.

---

### Code Breakdown

```python
# 1️ Initialize embedding layer weights using Xavier Uniform
nn.init.xavier_uniform_(self.embedding.weight)
```

 **Explanation**:

* `self.embedding` is the word embedding matrix of shape `[vocab_size, embedding_dim]`.
* `xavier_uniform_()` sets the weights to values from a **uniform distribution** in a range based on the number of input/output units.
* Xavier initialization is ideal when using **tanh** or **linear activations**.

 Purpose: Prevents activations from being too large or too small early in training.

---

```python
# 2️ Initialize LSTM weights and biases
for name, param in self.lstm.named_parameters():
    if 'weight' in name:
        nn.init.xavier_uniform_(param)
    elif 'bias' in name:
        nn.init.zeros_(param)
```

 **Explanation**:

* This loop iterates over **all parameters of LSTM**.
* LSTM has **multiple gates** (`input`, `forget`, `cell`, `output`) each with its own set of weights and biases.

🔹 Example LSTM parameter names:

```
weight_ih_l0  # input-hidden weights (input → gate)
weight_hh_l0  # hidden-hidden weights (hidden → gate)
bias_ih_l0    # bias for input-hidden
bias_hh_l0    # bias for hidden-hidden
```

🔹 `nn.init.xavier_uniform_(param)`: Properly initializes the weight matrices for **all gates**.

🔹 `nn.init.zeros_(param)`: Biases are initialized to **zero**. It's safe and often recommended.

---

```python
# 3️ Initialize Fully Connected Layer
nn.init.xavier_uniform_(self.fc.weight)
nn.init.zeros_(self.fc.bias)
```

 **Explanation**:

* `self.fc` is likely the **final linear layer**, usually projecting to the vocabulary or output classes.
* Again, we initialize weights smartly (Xavier) and zero the biases.

---

###  Recap Table

| Layer        | Type of Init   | Why?                                         |
| ------------ | -------------- | -------------------------------------------- |
| Embedding    | Xavier Uniform | Keeps word representations stable early on   |
| LSTM Weights | Xavier Uniform | Avoid exploding/vanishing gradients in gates |
| LSTM Biases  | Zeros          | Safe and simple to avoid adding bias early   |
| FC Weights   | Xavier Uniform | Ensures final projection is well-scaled      |
| FC Biases    | Zeros          | No bias unless learned                       |

---

###  When NOT to Use Xavier?

* Use **He Initialization** (`kaiming_`) if you're using **ReLU** activations.
* For **LSTMs**, Xavier is generally safe because tanh/sigmoid are used in gates.

