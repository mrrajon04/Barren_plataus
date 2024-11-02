import math
import torch
import csv
import os
import yaml
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
from plot import plot

with open('arguments.yaml', 'r') as file:
    args = yaml.safe_load(file)
csv_dir = "results/csv"
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
def train_model(model, X_train, y_train):
    if (args['method'] == 'beta' or args['method'] == 'uniform_norm' or args['method'] == 'classical_CNN'):
        model.train()
    num_epochs = args['epochs']
    barren_threshold = 1e-5

    with open(f"{csv_dir}/{args['method']}({args['dataset']}).csv", 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss', 'Gradient_norm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        if(args['method'] == 'beta' or args['method'] == 'uniform_norm' or args['method'] == 'classical_CNN'):
            for epoch in range(num_epochs):
                # Define loss function and optimizer
                optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
                optimizer.zero_grad()
                outputs = model(X_train)
                # print(outputs.shape, "shape")
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, y_train)
                loss.backward()

                # Calculate the average gradient norm for barren plateau detection
                total_norm = calculate_bp_gradient_norm(model)

                # Check if the gradient norm falls below the barren plateau threshold
                if total_norm < barren_threshold:
                    print(f"Barren Plateau detected at epoch {epoch + 1}: Gradient Norm = {total_norm:.8f}")

                optimizer.step()
                # Save loss and gradient norm to CSV file
                writer.writerow({
                    'Epoch': epoch + 1,
                    'Loss': loss.item(),
                    'Gradient_norm': total_norm
                })

                if epoch % 20 == 0 or epoch == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Gradient Norm: {total_norm:.8f}')
        elif(args['method'] == 'gaussian'):
            for epoch in range(num_epochs):
                loss = 0
                gradients_norm = 0
                for x, y in zip(X_train, y_train):
                    prediction = model.forward(x)
                    error = prediction - y

                    # Example of defining a QNode and computing gradients
                    # Define a cost function that takes both weights and inputs
                    def cost(weights):
                        return (model.circuit(weights, x) - y) ** 2  # Pass both weights and inputs

                    # Compute the gradients of the cost function with respect to weights
                    gradients = qml.jacobian(cost)(model.weights)
                    gradients_norms = calculate_gradient_norm(model)
                    gradients_norm = sum(gradients_norms) / len(gradients_norms)
                    model.update_weights(gradients, args['learning_rate'])

                    loss += (error ** 2)
                # Save loss and gradient norm to CSV file
                writer.writerow({
                    'Epoch': epoch + 1,
                    'Loss': loss.item(),
                    'Gradient_norm': gradients_norm
                })
                # Print the loss for each epoch
                if epoch % 20 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {loss: .6f} Gradients Norm: {gradients_norm:.6f}")
        elif(args['method'] == 'var_encoders'):
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

            for epoch in range(num_epochs):
                running_loss = 0
                optimizer.zero_grad()
                output = model(X_train)
                loss = torch.nn.CrossEntropyLoss()(output, y_train)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                gradient_norm = calculate_bp_gradient_norm(model)
                # Save loss and gradient norm to CSV file
                writer.writerow({
                    'Epoch': epoch + 1,
                    'Loss': running_loss,
                    'Gradient_norm': gradient_norm
                })
                if epoch % 20 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss} - Gradient Norm: {gradient_norm}")
        elif(args['method'] == 'time_nonlocal'):
            for epoch in range(100):
                model.optimizer.zero_grad()
                predictions = model.forward(X_train)
                loss = model.compute_loss(predictions, y_train)
                loss.backward()
                model.optimizer.step()
                gradient_norm = calculate_bp_gradient_norm(model)

                # Save loss and gradient norm to CSV file
                writer.writerow({
                    'Epoch': epoch + 1,
                    'Loss': loss.item(),
                    'Gradient_norm': gradient_norm
                })
                if epoch % 20 == 0 or epoch == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()} - Gradient Norm: {gradient_norm:.4f}")


    plot(f"results/csv/{args['method']}({args['dataset']}).csv")




def evaluate_model(model, X_test, y_test):
    accuracy = 0
    if(args['method'] == "gaussian"):
        correct = 0
        for x, y in zip(X_test, y_test):
            prediction = model.forward(x)
            if math.ceil(prediction) == y:
                correct += 1
        accuracy = correct / len(y_test)
    elif (args['method'] == "beta" or args['method'] == "uniform_norm" or args['method'] == "classical_CNN" or args['method'] == "time_nonlocal"):
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).float().mean()
    elif (args['method'] == "var_encoders"):
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()
        accuracy = correct/total
    elif(args['method'] == 'time_nonlocal'):
        model.train(X_test, y_test)
        with torch.no_grad():
            predictions = model(X_test)
            loss = model.compute_loss(predictions, y_test)
            accuracy = (predictions.argmax(dim=1) == y_test).float().mean().item()

            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

    # with open(f"{csv_dir}/{args['method']}({args['dataset']}).csv", 'w', newline='') as csvfile:
    #     fieldnames = ['accuracy']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     # writer.writeheader()
    #     writer.writerow({
    #         'accuracy': accuracy,
    #     })
    print(f'Accuracy: {accuracy:.4f}')
def calculate_gradient_norm(model):
    gradient_norms = []
    for weights in model.weights:
        norm = np.linalg.norm(weights)
        gradient_norms.append(norm)
    return gradient_norms
def calculate_bp_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
def calculate_loss(predictions, labels):
    return np.mean((predictions - labels) ** 2)
