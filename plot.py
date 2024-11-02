import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

with open('arguments.yaml', 'r') as file:
    args = yaml.safe_load(file)
def plot(csv):
    df = pd.read_csv(csv)

    ### Define colors for each dataset
    colors = ['royalblue', 'firebrick', 'goldenrod']  # Adjust colors as desired

    plt.figure(figsize=(8, 4))

    # Add a title for the whole figure
    title = f"{args['method']}({args['dataset']})"
    plt.suptitle(title, fontsize=10)

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Loss'], color=colors[0], linewidth=1, marker='o', markersize=1, label=args['dataset'])

    plt.title('Training Loss Over Epochs', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Training Loss', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(bottom=0)  # Adjust y-axis limit as needed
    plt.xlim(left=1)  # Set the minimum x-axis limit to 1
    plt.legend(loc="upper right", prop={'size': 10})  # Adjust legend position and font size

    # Gradient Norm Plot
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Gradient_norm'], color=colors[0], linewidth=1, marker='s', markersize=1, label=args['dataset'])
    plt.title('Gradient Norm Over Epochs', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Gradient Norm', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(bottom=0)  # Adjust y-axis limit as needed
    plt.xlim(left=1)  # Set the minimum x-axis limit to 1

    plt.legend(loc="upper right", prop={'size': 10})  # Adjust legend position and font size

    # Layout adjustment
    plt.tight_layout()

    # Create the "graphs" directory if it doesn't exist
    graphs_dir = "results/graphs"
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    # Save the plot with the title as the filename (replace '.png' with your preferred format)
    filename = os.path.join(graphs_dir, f"{args['method']}({args['dataset']}).png")
    plt.savefig(filename)

    print(f"Plot saved as: {filename}")

    plt.show()  # You can still display the plot after saving
