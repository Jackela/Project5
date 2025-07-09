#!/usr/bin/env python3
"""
generate_report.py

A flexible, command-line driven script to parse experiment artifacts, 
generate all necessary plots, and assemble a final project report.

Usage Example:
    python p5_text_classification/scripts/generate_report.py \\
        --exp "VDCNN-9 (Yelp)" REPORTS/logs/vdcnn/vdcnn_yelp_*.log REPORTS/vdcnn_yelp_predictions.csv \\
        --exp "VDCNN-9 (Yahoo)" REPORTS/logs/vdcnn/vdcnn_yahoo_*.log REPORTS/vdcnn_yahoo_predictions.csv \\
        --exp "DenseNet (Yelp)" REPORTS/logs/densenet/densenet_yelp_*.log REPORTS/densenet_yelp_predictions.csv \\
        --exp "DenseNet (Yahoo)" REPORTS/logs/densenet/densenet_yahoo_*.log REPORTS/densenet_yahoo_predictions.csv \\
        --exp "VDCNN-17 (Yahoo)" REPORTS/logs/vdcnn/vdcnn_yahoo_depth17_*.log REPORTS/vdcnn_yahoo_depth17_predictions.csv \\
        --output-dir REPORTS/plots
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import glob

# --- Configuration ---
# Define experiments and paths to their artifacts
# NOTE: Update log file names to the correct ones from your specific runs
# EXPERIMENTS = {
#     "densenet_yelp": {
#         "label": "DenseNet (Yelp)",
#         "preds": "REPORTS/densenet_yelp_predictions.csv",
#     },
#     "densenet_yahoo": {
#         "label": "DenseNet (Yahoo)",
#         "preds": "REPORTS/densenet_yahoo_predictions.csv",
#     },
#     "vdcnn_yelp_d9": {
#         "label": "VDCNN-9 (Yelp)",
#         "preds": "REPORTS/vdcnn_yelp_predictions.csv",
#     },
#     "vdcnn_yahoo_d9": {
#         "label": "VDCNN-9 (Yahoo)",
#         "preds": "REPORTS/vdcnn_yahoo_predictions.csv",
#     },
#     "vdcnn_yahoo_d17": {
#         "label": "VDCNN-17 (Yahoo)",
#         "preds": "REPORTS/vdcnn_yahoo_depth17_predictions.csv",
#     },
# }

# OUTPUT_DIR = Path("REPORTS/plots")

# --- Core Functions ---

def parse_log_file(log_path):
    """Parses a log file to extract epoch-by-epoch validation data and final metrics."""
    history = []
    text = log_path.read_text(encoding='utf-8')
    
    # Regex to find all epoch result blocks
    epoch_results = re.findall(
        r"EPOCH (\d+) RESULTS:.*?"
        r"Validation Loss: ([\d.]+).*?"
        r"Validation Accuracy: ([\d.]+)%",
        text,
        re.DOTALL
    )
    for epoch, val_loss, val_acc in epoch_results:
        history.append({
            'epoch': int(epoch),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc)
        })

    # Extract final metrics
    params_match = re.search(r'Trainable parameters: ([\d,]+)', text)
    time_match = re.search(r'Total training time: ([\d.]+) seconds', text)
    
    metrics = {
        'params': int(params_match.group(1).replace(',', '')) if params_match else 0,
        'time_min': float(time_match.group(1)) / 60 if time_match else 0
    }
    
    return pd.DataFrame(history), metrics

def find_latest_log_from_glob(pattern):
    """Finds the latest log file from a glob pattern."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return Path(max(files, key=lambda f: Path(f).stat().st_mtime))

def plot_individual_curves(df_history, label, output_dir):
    """Plots and saves validation loss and accuracy curves for a single experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")
    
    plot_label = label.replace(" ", "_").lower()
    
    # Accuracy
    sns.lineplot(x='epoch', y='val_accuracy', data=df_history, ax=ax1, label='Validation Accuracy')
    ax1.set_title(f'Validation Accuracy: {label}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    
    # Loss
    sns.lineplot(x='epoch', y='val_loss', data=df_history, ax=ax2, label='Validation Loss', color='tab:red')
    ax2.set_title(f'Validation Loss: {label}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{plot_label}_curves.png", dpi=300)
    plt.close()

def plot_comparison(data, x, y, hue, title, output_path, kind='bar', palette='viridis', rotation=0):
    """A generic plotting function for comparisons."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    if kind == 'bar':
        ax = sns.barplot(x=x, y=y, hue=hue, data=data, palette=palette)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
    elif kind == 'line':
        ax = sns.lineplot(x=x, y=y, hue=hue, data=data, palette=palette, marker='o', linestyle='-')
    
    plt.title(title, fontsize=16)
    if rotation:
        plt.xticks(rotation=rotation, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {output_path}")

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate a full report from experiment artifacts.")
    parser.add_argument(
        '--exp',
        nargs=3,
        action='append',
        metavar=('LABEL', 'LOG_GLOB', 'PREDS_PATH'),
        required=True,
        help="Define an experiment: Label, glob pattern for log file, path to predictions CSV. Can be used multiple times."
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('REPORTS/plots'),
        help="Directory to save all generated plots."
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    experiments = {
        f"exp_{i}": {
            "label": exp_args[0],
            "log_glob": exp_args[1],
            "preds": exp_args[2]
        }
        for i, exp_args in enumerate(args.exp)
    }

    all_metrics = []

    print("--- Parsing logs and generating individual plots ---")
    for name, config in experiments.items():
        print(f"Processing: {config['label']}")
        log_path = find_latest_log_from_glob(config['log_glob'])
        if not log_path:
            print(f"  -> ⚠️ Log file not found for glob: {config['log_glob']}. Skipping.")
            continue
        
        df_history, metrics = parse_log_file(log_path)
        plot_individual_curves(df_history, config['label'], OUTPUT_DIR)
        
        # Aggregate metrics for comparison plots
        df_preds = pd.read_csv(config['preds'])
        accuracy = (df_preds['true_label'] == df_preds['predicted_label']).mean() * 100
        all_metrics.append({
            'name': name,
            'label': config['label'],
            'accuracy': accuracy,
            **metrics
        })
    df_metrics = pd.DataFrame(all_metrics)

    # --- 2. Generate Core Comparison Plot ---
    print("\n--- Generating core comparison plot ---")
    # This part needs to be dynamic based on the provided experiments
    # For now, we'll just generate a plot for all unique labels found in the provided experiments
    unique_labels = df_metrics['label'].unique()
    if len(unique_labels) > 0:
        plot_comparison(
            data=df_metrics, x='label', y='accuracy', hue=None,
            title='Core Experiments: Final Accuracy Comparison',
            output_path=OUTPUT_DIR / 'core_accuracy_comparison.png',
            rotation=15
        )
    else:
        print("No unique labels found for core comparison plot.")

    # --- 3. Generate Bonus (Depth) Comparison Plots ---
    print("\n--- Generating bonus (depth) comparison plots ---")
    # This part needs to be dynamic based on the provided experiments
    # We'll identify experiments that are likely depth variations
    depth_experiments = []
    for name, config in experiments.items():
        if 'vdcnn' in name and 'd9' in name:
            depth_experiments.append(name)
        elif 'vdcnn' in name and 'd17' in name:
            depth_experiments.append(name)
    
    if depth_experiments:
        depth_metrics = df_metrics[df_metrics['label'].isin([experiments[name]['label'] for name in depth_experiments])]
        
        # Reshape for multi-metric plot
        depth_metrics_melted = depth_metrics.melt(
            id_vars='label', 
            value_vars=['accuracy', 'params', 'time_min'],
            var_name='Metric', 
            value_name='Value'
        )
        # Nicer labels for plot
        depth_metrics_melted['Metric'] = depth_metrics_melted['Metric'].replace({
            'accuracy': 'Accuracy (%)',
            'params': 'Parameters (Millions)',
            'time_min': 'Train Time (min)'
        })
        # Scale params
        depth_metrics_melted.loc[depth_metrics_melted['Metric'] == 'Parameters (Millions)', 'Value'] /= 1_000_000

        g = sns.catplot(x='Metric', y='Value', hue='label', data=depth_metrics_melted,
                        kind='bar', palette='muted', height=6, aspect=1.5)
        g.fig.suptitle('VDCNN Depth Comparison: Performance vs. Cost', y=1.02, fontsize=16)
        g.set_axis_labels("Metric", "Value")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(OUTPUT_DIR / 'depth_tradeoff_comparison.png', dpi=300)
        plt.close()
        print(f"✅ Saved plot: {OUTPUT_DIR / 'depth_tradeoff_comparison.png'}")
        
        # Comparative accuracy curves
        history_dfs = []
        for name in depth_experiments:
            log_path = find_latest_log_from_glob(experiments[name]['log_glob'])
            if not log_path and 'd17' in name: # Special case for d17 if name differs
                log_path = find_latest_log_from_glob(f"REPORTS/logs/vdcnn/vdcnn_yahoo_depth17_*.log")

            if log_path:
                df_hist, _ = parse_log_file(log_path)
                df_hist['label'] = experiments[name]['label']
                history_dfs.append(df_hist)
        
        if history_dfs:
            plot_comparison(
                data=pd.concat(history_dfs), x='epoch', y='val_accuracy', hue='label',
                title='VDCNN Depth: Validation Accuracy Curves',
                output_path=OUTPUT_DIR / 'depth_accuracy_curves.png',
                kind='line'
            )
    else:
        print("No depth experiments found for comparative accuracy curves.")
    
    print("\nAll plots generated successfully.")

if __name__ == "__main__":
    main() 