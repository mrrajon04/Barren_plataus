import torch
import numpy as np
import pennylane.numpy as qnp
from scipy.stats import truncnorm, beta
import scipy.stats as ss

import yaml

with open('arguments.yaml', 'r') as file:
    args = yaml.safe_load(file)

### initializations ###

def init_beta_ebayes(data, shape):
    data = data.cpu().detach().numpy().flatten()  # Flatten for normalization
    data_r = data.reshape(data.size)
    sz = np.prod(shape)

    dmin, dmax = np.min(data_r), np.max(data_r)
    for i in range(len(data_r)):
        data_r[i] =  (data_r[i] - dmin) / (dmax - dmin)
    zmask = data_r <= 0
    omask = data_r >= 1
    data_r[zmask] = data_r[zmask] + 1e-8
    data_r[omask] = data_r[omask] - 1e-8
    print(data_r, "dr")

    def filter_array(data, min, max):
        # Filter the array based on the condition
        return [x for x in data_r if 0 < (x - dmin) / (dmax - dmin) < 1]

    filtered_data = filter_array(data_r, dmin, dmax)

    a, b, _, _ = beta.fit(filtered_data, floc=0, fscale=1)
    print(f"Found alpha:{a}, beta:{b}")
    return qnp.random.beta(a=a, b=b, size=sz).reshape(shape)

def init_uniform_norm(data, shape): # Implemented
    data = data.cpu().detach().numpy().flatten()  # Flatten for normalization
    dr = data.reshape(data.size)
    sz = np.prod(shape)
    dmin, dmax = np.min(data), np.max(data)

    for i in range(len(data)):
        dr[i] = (dr[i] - dmin)/(dmax - dmin)
    zmask = dr <= 0
    omask = dr >= 1
    dr[zmask] = dr[zmask] + 1e-8
    dr[omask] = dr[omask] - 1e-8
    l, h = ss.uniform.fit(dr)
    print(f"Determined range: [{l},{h}]")
    return qnp.random.uniform(l, h, size=sz).reshape(shape)
# Gaussian_initialization
def gaussian_initialization(num_qubits, layers):
    # Using Gaussian distribution N(0, σ^2) for initialization
    sigma = 1 / (2 * layers)
    weights = np.random.normal(0, sigma, (layers, num_qubits, 2))
    return weights
# Variational_encoders
def initialize_params(model):
    # Example of a custom initialization strategy to prevent barren plateaus
    for param in model.parameters():
        torch.nn.init.uniform_(param, a=-0.1, b=0.1)
# time_nonlocal
class FourierInitialization:
    def initialize_params(self):
        # Implement Fourier initialization logic
        params = torch.randn(4, 784)  # Example: Random 4x4 matrix for Iris datasets
        # params = torch.randn(784, 4)  # Example: Random 4x4 matrix for Mnist datasets
        return torch.nn.Parameter(params)

class StepwiseInitialization:
    def initialize_params(self):
        # Implement stepwise initialization logic
        params = torch.randn(4, 784)  # Example: Random 4x4 matrix for Iris datasets
        # params = torch.randn(784, 4)  # Example: Random 4x4 matrix for Mnist datasets
        return torch.nn.Parameter(params)

def CNNInit(layer):
    """initialization for linear layers."""
    if isinstance(layer, torch.nn.Linear):
        with torch.no_grad():  # Disable gradient tracking for weight initialization
            # initialization for "init_uniform_norm" from avoiding barren plateaus algorithm
            if(args['method'] == 'classical_CNN'):
                torch.nn.init.uniform_(layer.weight)
            else:
                if(args['method'] == 'beta'):
                    dta = torch.from_numpy(init_beta_ebayes(layer.weight.data, layer.weight.data.shape))
                elif(args['method'] == 'uniform_norm'):
                    dta = torch.from_numpy(init_uniform_norm(layer.weight.data, layer.weight.data.shape))
                dta = dta.float() # float32
                layer.weight.data = dta;
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)