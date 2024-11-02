import torch
import torch.nn as nn
import pennylane as qml
import torch.optim as optim
from pennylane import numpy as np
import yaml
#
with open('arguments.yaml', 'r') as file:
    args = yaml.safe_load(file)
#
# Define model parameters
hidden_dim = 10
output_dim = 10
if args['dataset'] == "Iris":
    output_dim = 3
class IrisModel(nn.Module):
    def __init__(self, input_dim, init_fn=None):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # Apply custom initialization if provided
        if init_fn:
            self.apply(init_fn)

    def forward(self, x):
        y = self;
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Gaussian
class QuantumModel:
    def __init__(self, initialization_strategy, num_qubits=4, layers=5):
        self.num_qubits = num_qubits
        self.layers = layers

        self.init_strategy = initialization_strategy
        self.weights = np.array(self.init_strategy(self.num_qubits, self.layers), requires_grad=True)

        # Define a quantum device
        self.device = qml.device('default.qubit', wires=num_qubits)

        @qml.qnode(self.device)
        def circuit(weights, inputs):
            # Quantum circuit definition
            for layer in range(layers):
                for i in range(num_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                qml.CZ(wires=[i, (i + 1) % num_qubits])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, inputs):
        return self.circuit(self.weights, inputs)

    def update_weights(self, gradients, lr):
        self.weights -= lr * gradients
# Variational_encoders
class QuantumNeuralNetwork(nn.Module):
    def __init__(self):
        super(QuantumNeuralNetwork, self).__init__()
        self.n_qubits = 4  # Example
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weights = np.random.rand(4, 4, 3)
        self.circuit = circuit
        self.weight_shapes = {"weights": weights}  # Example
        if(args['dataset'] == "Iris"):
            self.fc = nn.Linear(self.n_qubits, 3)  # 3 output classes for Iris
        elif(args['dataset'] == "MNIST" or args['dataset'] == "MedMNIST"):
            self.fc = nn.Linear(self.n_qubits, 10)  # 3 output classes for mnist

    def forward(self, x):
        batch_size = x.shape[0]  # Get the batch size (120 in your case)
        q_out_list = []

        # Apply the quantum circuit for each sample in the batch
        for i in range(batch_size):
            q_out = self.circuit(x[i], self.weight_shapes["weights"])
            q_out_list.append(q_out)

        # Convert the list of quantum outputs to a tensor
        q_out_tensor = torch.tensor(q_out_list, dtype=torch.float32)  # Shape: [120, 4]
        # print(q_out_tensor.shape, "from forward")
        # Pass through the fully connected layer (applied to each sample independently)
        output = self.fc(q_out_tensor)  # Shape: [120, 3]

        return output
# time_nonlcal
class QuantumModel2(nn.Module):
    def __init__(self, initialization_strategy):
        super(QuantumModel2, self).__init__()
        self.params = initialization_strategy.initialize_params()
        self.optimizer = optim.Adam([self.params], lr=args['learning_rate'])
        if(args['dataset'] == 'MNIST'):
            self.fc = nn.Linear(784, 10)  # Change 4 to 784 to match the MNIST input size

    #
    def forward(self, X):
        # Apply quantum operations here
        if args['dataset'] == 'MNIST':
            # Reshape X to [batch_size, 784] if it's in [batch_size, 1, 28, 28]
            if X.dim() > 2:
                X = X.view(X.size(0), -1)  # Flatten the input
            # Perform forward pass using the fully connected layer for MNIST
            output = self.fc(X)  # Shape [batch_size, 10]
            return output  # Output shape should be [batch_size, 10]

        elif(args['dataset'] == 'Iris' or args['dataset'] == 'MedMNIST'):
            return (X.mm(self.params))
    def compute_loss(self, predictions, targets):
        criterion = nn.CrossEntropyLoss()
        return criterion(predictions, targets)
