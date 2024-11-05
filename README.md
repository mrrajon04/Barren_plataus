# Benchmark_BP_Free

Benchmark_BP_Free is a research framework for benchmarking. This repository allows experimentation with various Initialization, Model and Optimization methods and training configurations across different datasets.

---

## Installation

To install the framework from source, follow these steps:

1. Clone the repository:
   
   ```bash
   git clone https://github.com/mrrajon04/Benchmark_BP_Free.git
3. Navigate to the project directory:
```
cd Benchmark_BP_Free
```
# Usage
Train model
This framework supports training on multiple datasets using different initialization methods. Below is an example of training a model with Gaussian initialization on the MNIST dataset for 20 epochs.

1. Configure Training Parameters
Update the arguments.yaml file with the desired settings:


    Example: Train a Gaussian Initilization on MNIST dataset for 20 epoch.
   On arguements.yaml
    ```python
    method: 'gaussian'
    dataset: 'MNIST'
    epochs: 20
    barren_threshold: 1e-5
    learning_rate: 0.01 # Possible Learning_rate: 0.0001 to 0.1
    ```
2. Run Training
   After configuring the arguments.yaml file, start training by running:


   ```
   python main.py
