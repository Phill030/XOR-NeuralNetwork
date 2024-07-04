<div align="center">
  <h1>NeuralNetwork<br></br>
    <a href="https://ko-fi.com/phill030">
      <img alt="Kofi" src="https://shields.io/badge/Kofi-Buy_me_a_coffee-ff5f5f?logo=ko-fi&style=for-the-badgeKofi">
    </a>
  </h1>
</div>

- [Getting Started](#getting-started)
- [Credits](#credits)
- [Features](#features)
- [Usage](#usage)

---

# Getting Started
This project implements a basic neural network from scratch in Rust. It includes an input layer, hidden layers, and an output layer, allowing for flexible configuration and training. The neural network uses the sigmoid activation function and supports backpropagation for training. Additionally, the project includes functionality to visualize the training loss over epochs using the plotters crate.

<div align="center">
  <img width="600px" src="https://github.com/Phill030/XOR-NeuralNetwork/assets/50775241/c847c239-070c-4643-9aa9-f290fedfed94" />
</div>

# Credits
Thanks to the help of [@FrozenAssassine](https://github.com/FrozenAssassine) and [@finn-freitag](https://github.com/finn-freitag/) I've built this NeuralNetwork from scratch!

# Features
- **Layered Architecture:** Includes input, hidden, and output layers.
- **Feed Forward:** Implemented feed-forward algorithm.
- **Backpropagation:** Training through backpropagation using gradient descent.
- **Cross-Entropy Loss:** Utilized cross-entropy loss for binary classification.
- **Visualization:** Plot training loss over epochs using plotters library.


# Usage
1. Clone the repository to your local machine:
```bash
git clone https://github.com/Phill030/XOR-NeuralNetwork.git
cd XOR-NeuralNetwork
```

2. Build the project using Cargo:
```bash
cargo run
```

This will train the neural network on a simple XOR dataset, print the predictions, and generate a plot of the training loss over epochs as loss_plot.png.
