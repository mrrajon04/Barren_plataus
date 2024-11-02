from colorama import Fore, Style
from init_strat import CNNInit
from model import QuantumModel, QuantumModel2,  IrisModel,QuantumNeuralNetwork

from init_strat import gaussian_initialization, initialize_params,FourierInitialization
from load_data import load_iris_data, load_mnist_data
from utils import train_model, evaluate_model
import yaml

import warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

with open('arguments.yaml', 'r') as file:
    args = yaml.safe_load(file)

print(f"{Fore.CYAN}{Style.BRIGHT}Configuration Settings for Model Training{Style.RESET_ALL}")
print(f"{Fore.YELLOW}-----------------------------------------{Style.RESET_ALL}")
print(f"{Fore.GREEN}Training Method: {Style.RESET_ALL}{Fore.WHITE}{args['method'].capitalize()}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Dataset: {Style.RESET_ALL}{Fore.WHITE}{args['dataset'].upper()}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Total Epochs: {Style.RESET_ALL}{Fore.WHITE}{args['epochs']}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Learning Rate: {Style.RESET_ALL}{Fore.WHITE}{args['learning_rate']}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Barren Plateau Threshold: {Style.RESET_ALL}{Fore.WHITE}{args['barren_threshold']}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}-----------------------------------------{Style.RESET_ALL}")
print(f"{Fore.CYAN}Ensure all settings align with the model requirements.{Style.RESET_ALL}")


def main():

    # Load data
    if(args['dataset'] == "MNIST"):
        X_train, X_test, y_train, y_test = load_mnist_data()
    elif(args['dataset'] == "Iris"):
        X_train, X_test, y_train, y_test = load_iris_data()
    elif (args['dataset'] == "MedMNIST"):
        X_train, X_test, y_train, y_test = load_iris_data()

    if(args['method'] == "beta" or args['method'] == "uniform_norm" or args['method'] == "classical_CNN"):
        # Initialize the model with CNNInit strategy
        model = IrisModel(X_train.shape[1], init_fn=CNNInit)

    elif(args['method'] == "gaussian"):
        # Initialize the model with Gaussian strategies
        model = QuantumModel(initialization_strategy=gaussian_initialization)

    elif(args['method'] == "var_encoders"):
        # Initialize model
        model = QuantumNeuralNetwork()
        # Initialize strategies
        initialize_params(model)
    elif(args['method'] == "time_nonlocal"):
        # Choose Initialization Strategy (Fourier or Stepwise)
        init_strategy = FourierInitialization()  # Switch to StepwiseInitialization() if needed
        # Initialize Quantum Model
        model = QuantumModel2(init_strategy)



    # Train the model and check for barren plateaus (gradient norms)
    # Train the model
    train_model(model, X_train, y_train)
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
