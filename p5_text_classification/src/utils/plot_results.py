import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_training_results(results_path: Path):
    """
    Reads a training history CSV file and generates plots for accuracy and loss.

    Args:
        results_path (Path): Path to the training history CSV file.
    """
    if not results_path.is_file():
        print(f"Error: Results file not found at {results_path}")
        return

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training & Validation Performance', fontsize=16)

    # Plot Accuracy
    sns.lineplot(x=df.index, y=df['train_accuracy'], ax=ax1, label='Train Accuracy', color='tab:blue', lw=2)
    sns.lineplot(x=df.index, y=df['val_accuracy'], ax=ax1, label='Validation Accuracy', color='tab:orange', lw=2)
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.set_ylim(min(df['train_accuracy'].min(), df['val_accuracy'].min()) * 0.95, 1.0) # Adjust y-axis

    # Plot Loss
    sns.lineplot(x=df.index, y=df['train_loss'], ax=ax2, label='Train Loss', color='tab:blue', lw=2)
    sns.lineplot(x=df.index, y=df['val_loss'], ax=ax2, label='Validation Loss', color='tab:orange', lw=2)
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plot_path = results_path.with_suffix('.png')
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Plot saved to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training results from a CSV file.")
    parser.add_argument('--results_file', type=str, required=True,
                        help="Path to the training history CSV file.")
    args = parser.parse_args()
    
    plot_training_results(Path(args.results_file)) 