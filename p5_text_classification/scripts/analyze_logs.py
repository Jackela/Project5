#!/usr/bin/env python3
"""
Training Log Analysis and Report Generation Script

This script analyzes training logs and CSV files from experiments,
generating comprehensive reports with visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix

# Determine project root (workspace root) dynamically three levels up
project_root = Path(__file__).resolve().parent.parent.parent

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Configure Chinese font support for potential localized content in plots
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Configure professional plotting style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class ExperimentAnalyzer:
    """
    Analyzer for training experiment logs and results.
    
    Responsibilities:
    - Parse training logs and CSV files
    - Extract key metrics and metadata
    - Generate performance visualizations
    - Create comprehensive reports
    """
    
    def __init__(self, reports_dir: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            reports_dir: Path to the main reports directory. If None, it will be
                         dynamically located at {project_root}/reports.
        """
        if reports_dir:
            self.reports_dir = Path(reports_dir)
        else:
            self.reports_dir = project_root / "reports"
            
        self.experiments: List[Dict] = []
        
    def discover_experiments(self) -> List[Dict]:
        """
        Discover all completed experiments from log files.
        
        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []
        
        # Define paths relative to the robustly determined reports_dir
        log_dir = self.reports_dir / "logs"
        results_dir = self.reports_dir / "results"
        history_dir = self.reports_dir / "training_history"

        if not log_dir.is_dir():
            print(f"Warning: Log directory not found at {log_dir}")
            return []

        for log_file in log_dir.glob("*.log"):
            # Parse filename: model_dataset_timestamp.log
            match = re.match(r'(\w+)_(\w+)_(\d{8}_\d{6})\.log', log_file.name)
            if match:
                model, dataset, timestamp = match.groups()
                
                # Look for corresponding files
                csv_file = history_dir / f"{model}_{dataset}_{timestamp}.csv"
                results_file = results_dir / f"{model}_{dataset}_{timestamp}_results.csv"

                experiment = {
                    'model': model,
                    'dataset': dataset,
                    'timestamp': timestamp,
                    'log_file': log_file,
                    'csv_file': csv_file if csv_file.exists() else None,
                    'results_file': results_file if results_file.exists() else None,
                    'completed': self._check_completion(log_file)
                }
                
                if experiment['completed']:
                    experiments.append(experiment)
        
        self.experiments = sorted(experiments, key=lambda x: x['timestamp'], reverse=True)
        return self.experiments
    
    def _check_completion(self, log_file: Path) -> bool:
        """Check if the framework execution completed successfully."""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Consider both old and new success markers
                return any(marker in content for marker in [
                    "FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY",
                    "✅ Experiment finished",
                    "✅ Completed",
                    "🏁 TRAINING FINISHED"
                ])
        except Exception:
            return False
    
    def parse_log_metrics(self, log_file: Path) -> Dict:
        """
        Parse training metrics from log file.
        
        Args:
            log_file: Path to log file
            
        Returns:
            Dictionary containing parsed metrics
        """
        metrics = {
            'final_accuracy': None,
            'final_loss': None,
            'epochs_completed': 0,
            'model_parameters': None,
            'device': None,
            'batch_size': None,
            'learning_rate': None,
            'training_samples': None,
            'validation_samples': None
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract final accuracy
            acc_match = re.search(r'Final validation accuracy: ([\d.]+)%', content)
            if acc_match:
                metrics['final_accuracy'] = float(acc_match.group(1))
            
            # Extract model parameters
            param_match = re.search(r"Model '\w+' created with ([\d,]+) parameters", content)
            if param_match:
                metrics['model_parameters'] = int(param_match.group(1).replace(',', ''))
            
            # Extract device info
            device_match = re.search(r'Device: (\w+)', content)
            if device_match:
                metrics['device'] = device_match.group(1)
            
            # Extract training configuration
            batch_match = re.search(r'Batch Size: (\d+)', content)
            if batch_match:
                metrics['batch_size'] = int(batch_match.group(1))
            
            lr_match = re.search(r'Learning Rate: ([\d.e-]+)', content)
            if lr_match:
                metrics['learning_rate'] = float(lr_match.group(1))
            
            # Extract data sizes
            data_match = re.search(r'Data loaded: (\d+) train, (\d+) validation samples', content)
            if data_match:
                metrics['training_samples'] = int(data_match.group(1))
                metrics['validation_samples'] = int(data_match.group(2))
            
            # Count epochs
            epoch_matches = re.findall(r'EPOCH (\d+)/(\d+)', content)
            if epoch_matches:
                metrics['epochs_completed'] = len(set(match[0] for match in epoch_matches))
                
        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")
        
        return metrics
    
    def load_training_history(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """Load training history from CSV file."""
        if not csv_file or not csv_file.exists():
            return None
        
        try:
            df = pd.read_csv(csv_file)
            # Add epoch column if missing
            if 'epoch' not in df.columns:
                df['epoch'] = range(1, len(df) + 1)
            return df
        except Exception as e:
            print(f"Error loading CSV file {csv_file}: {e}")
            return None
    
    def generate_performance_plots(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        Generate performance visualization plots.
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir is None:
            output_dir = self.reports_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # 1. Model Comparison - Final Accuracy
        self._plot_model_comparison(output_dir)
        plots['model_comparison'] = output_dir / "model_comparison.png"
        
        # 2. Parameter Count Comparison
        self._plot_parameter_comparison(output_dir)
        plots['parameter_comparison'] = output_dir / "parameter_comparison.png"
        
        # 3. Training Curves (if CSV data available)
        self._plot_training_curves(output_dir)
        plots['training_curves'] = output_dir / "training_curves.png"

        # 4. Confusion Matrices
        self._plot_all_confusion_matrices(output_dir)
        plots['confusion_matrices'] = output_dir # This is a directory
        
        return plots
    
    def _plot_model_comparison(self, output_dir: Path):
        """Plot model performance comparison."""
        data = []
        for exp in self.experiments:
            metrics = self.parse_log_metrics(exp['log_file'])
            if metrics['final_accuracy'] is not None:
                data.append({
                    'Model': exp['model'].upper(),
                    'Dataset': exp['dataset'].upper(),
                    'Accuracy': metrics['final_accuracy'],
                    'Experiment': f"{exp['model']}_{exp['dataset']}"
                })
        
        if not data:
            print("No accuracy data found for plotting")
            return
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar plot
        ax = sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model', 
                        palette=['#2E86C1', '#E74C3C'])
        
        plt.title('Model Performance Comparison\nValidation Accuracy by Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Dataset', fontsize=14, fontweight='bold')
        plt.ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', fontweight='bold')
        
        plt.legend(title='Model', title_fontsize=12, fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_class_names(self, dataset_name: str) -> Optional[List[str]]:
        """Load class names from the dataset's classes.txt file."""
        # Use dynamic project_root to locate data directory
        data_dir = project_root / "data" 
        class_file = data_dir / dataset_name / "classes.txt"
        if not class_file.exists():
            print(f"Warning: Class file not found for dataset '{dataset_name}' at {class_file}")
            return None
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                # Class labels are 1-based, but list is 0-indexed
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error reading class file {class_file}: {e}")
            return None

    def _plot_all_confusion_matrices(self, output_dir: Path):
        """Generate and save confusion matrix for each experiment."""
        cm_dir = output_dir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)

        for exp in self.experiments:
            if exp.get('results_file') and exp['results_file'].exists():
                try:
                    results_df = pd.read_csv(exp['results_file'])
                    y_true = results_df['true_label']
                    y_pred = results_df['predicted_label']
                    
                    class_names = self._get_class_names(exp['dataset'])
                    if class_names is None:
                        # Generate generic labels if class file is missing
                        num_classes = max(y_true.max(), y_pred.max()) + 1
                        class_names = [f"Class {i}" for i in range(num_classes)]

                    cm = confusion_matrix(y_true, y_pred)
                    
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=class_names, yticklabels=class_names)
                    
                    plt.title(f"Confusion Matrix - {exp['model'].upper()} on {exp['dataset'].upper()}", 
                              fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
                    plt.ylabel("True Label", fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45, ha="right")
                    plt.yticks(rotation=0)
                    
                    filename = f"cm_{exp['model']}_{exp['dataset']}_{exp['timestamp']}.png"
                    save_path = cm_dir / filename
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Add path to experiment dict for report generation
                    exp['confusion_matrix_plot'] = save_path

                except Exception as e:
                    print(f"Error generating confusion matrix for {exp['log_file'].name}: {e}")

    def _plot_parameter_comparison(self, output_dir: Path):
        """Plot model parameter count comparison, correctly grouped by model type."""
        data = []
        # Use a dictionary to store unique models and their parameter counts
        model_params = {}
        for exp in self.experiments:
            metrics = self.parse_log_metrics(exp['log_file'])
            model_name = exp['model'].upper()
            if metrics['model_parameters'] is not None and model_name not in model_params:
                model_params[model_name] = metrics['model_parameters']

        # Convert the dictionary to a list of dicts for DataFrame creation
        for model, params in model_params.items():
            data.append({'Model': model, 'Parameters': params})

        if not data:
            print("No model parameter data found for plotting")
            return
        
        df = pd.DataFrame(data)
        df['Parameters (Millions)'] = df['Parameters'] / 1_000_000
        
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(data=df, x='Model', y='Parameters (Millions)', 
                         palette=['#E74C3C', '#2E86C1'])
        
        plt.title('Model Complexity Comparison\nParameter Count', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.ylabel('Parameters (Millions)', fontsize=14, fontweight='bold')
        
        # Add value labels on bars, formatted in millions
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2fM', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "parameter_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_training_curves(self, output_dir: Path):
        """Plot training and validation curves for each experiment."""
        plt.figure(figsize=(15, 10))
        
        colors = {'vdcnn': '#2E86C1', 'densenet': '#E74C3C'}
        linestyles = {'yelp': '-', 'yahoo': '--'}
        
        # Create subplots for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        plot_data = []
        
        for exp in self.experiments:
            if exp['csv_file'] and exp['csv_file'].exists():
                df = self.load_training_history(exp['csv_file'])
                if df is not None and len(df) > 1:
                    label = f"{exp['model'].upper()}_{exp['dataset'].upper()}"
                    color = colors.get(exp['model'], '#666666')
                    linestyle = linestyles.get(exp['dataset'], '-')
                    
                    # Plot training and validation loss
                    if 'train_loss' in df.columns:
                        ax1.plot(df['epoch'], df['train_loss'], 
                               color=color, linestyle=linestyle, alpha=0.7,
                               label=f"{label}_train_loss")
                    if 'val_loss' in df.columns:
                        ax1.plot(df['epoch'], df['val_loss'], 
                               color=color, linestyle=linestyle, linewidth=2,
                               label=f"{label}_val_loss")
                    
                    # Plot validation accuracy
                    if 'val_accuracy' in df.columns:
                        ax2.plot(df['epoch'], df['val_accuracy'], 
                               color=color, linestyle=linestyle, linewidth=2,
                               label=label)
                    
                    plot_data.append(exp)
        
        if plot_data:
            # Configure loss plot
            ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Configure accuracy plot
            ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        else:
            print("No training curve data found")
        
        plt.close()
    
    def generate_report(self, output_file_name: str = "experiment_analysis.md") -> Path:
        """
        Generate comprehensive experiment report in the correct reports directory.
        
        Args:
            output_file_name: The name of the output report file.
        
        Returns:
            Path to generated report file
        """
        # Ensure the final report is created inside the correct, unified reports directory
        report_path = self.reports_dir / output_file_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate plots first, ensuring they also go into the correct subdirectory
        plots_dir = self.reports_dir / "plots"
        plots = self.generate_performance_plots(plots_dir)
        
        # Create report content
        report_content = self._create_report_content(plots)
        
        # Write report
        report_path.write_text(report_content, encoding='utf-8')
        
        print(f"✅ Comprehensive report generated: {report_path}")
        return report_path

    def _create_report_content(self, plots: Dict[str, Path]) -> str:
        """Create the main report content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Deep Learning Text Classification - Experiment Report

**Generated on:** {timestamp}
**Total Experiments:** {len(self.experiments)}

## Executive Summary

This report presents the results of our comprehensive text classification experiments using two state-of-the-art deep learning architectures: VDCNN (Very Deep Convolutional Networks) and DenseNet-BC (Densely Connected Convolutional Networks with Bottleneck and Compression).

### Key Findings

"""
        
        # Add experiment overview
        report += self._add_experiment_overview()
        
        # Add detailed results
        report += "\n## Detailed Experiment Results\n\n"
        report += self._add_detailed_results()
        
        # Add visualizations
        if plots:
            report += "\n## Performance Visualizations\n\n"
            
            # Model Comparison Plot
            if 'model_comparison' in plots and plots['model_comparison'].exists():
                relative_path = Path('plots') / plots['model_comparison'].name
                report += f"### Model Performance Comparison\n![Model Comparison]({relative_path})\n"

            # Parameter Comparison Plot
            if 'parameter_comparison' in plots and plots['parameter_comparison'].exists():
                relative_path = Path('plots') / plots['parameter_comparison'].name
                report += f"\n### Model Complexity Comparison\n![Parameter Comparison]({relative_path})\n"

            # Training Curves Plot
            if 'training_curves' in plots and plots['training_curves'].exists():
                relative_path = Path('plots') / plots['training_curves'].name
                report += f"\n### Training & Validation Curves\n![Training Curves]({relative_path})\n"

            # Confusion Matrices
            report += "\n### Confusion Matrices\n"
            cm_plot_found = False
            for exp in self.experiments:
                if 'confusion_matrix_plot' in exp and exp['confusion_matrix_plot'].exists():
                    cm_plot_found = True
                    relative_path = Path('plots/confusion_matrices') / exp['confusion_matrix_plot'].name
                    report += f"\n#### {exp['model'].upper()} on {exp['dataset'].upper()}\n"
                    report += f"![CM {exp['model']} {exp['dataset']}]({relative_path})\n"
            
            if not cm_plot_found:
                report += "No confusion matrix data available.\n"
        
        # Add conclusions
        report += self._add_conclusions()
        
        return report
    
    def _add_experiment_overview(self) -> str:
        """Add experiment overview section."""
        overview = ""
        
        # Group experiments by model and dataset
        summary_data = {}
        for exp in self.experiments:
            key = f"{exp['model']}_{exp['dataset']}"
            metrics = self.parse_log_metrics(exp['log_file'])
            summary_data[key] = metrics
        
        if summary_data:
            overview += "| Experiment | Final Accuracy | Parameters | Training Samples |\n"
            overview += "|------------|---------------|------------|------------------|\n"
            
            for exp_name, metrics in summary_data.items():
                # Handle complex experiment names by taking first two parts after underscore split
                parts = exp_name.split('_')
                if len(parts) >= 2:
                    model, dataset = parts[0], parts[1]
                else:
                    model, dataset = exp_name, "unknown"
                    
                acc = f"{metrics['final_accuracy']:.1f}%" if metrics['final_accuracy'] else "N/A"
                params = f"{metrics['model_parameters']:,}" if metrics['model_parameters'] else "N/A"
                samples = f"{metrics['training_samples']}" if metrics['training_samples'] else "N/A"
                
                overview += f"| {model.upper()} on {dataset.upper()} | {acc} | {params} | {samples} |\n"
        
        return overview
    
    def _add_detailed_results(self) -> str:
        """Add detailed results for each experiment."""
        details = ""
        
        for exp in self.experiments:
            metrics = self.parse_log_metrics(exp['log_file'])
            
            details += f"### {exp['model'].upper()} on {exp['dataset'].upper()} Dataset\n\n"
            details += f"**Experiment ID:** {exp['timestamp']}  \n"
            details += f"**Log File:** `{exp['log_file'].name}`  \n"
            
            if metrics['final_accuracy']:
                details += f"**Final Validation Accuracy:** {metrics['final_accuracy']:.2f}%  \n"
            
            if metrics['model_parameters']:
                details += f"**Model Parameters:** {metrics['model_parameters']:,}  \n"
            
            if metrics['epochs_completed']:
                details += f"**Epochs Completed:** {metrics['epochs_completed']}  \n"
            
            if metrics['device']:
                details += f"**Training Device:** {metrics['device']}  \n"
            
            if metrics['batch_size']:
                details += f"**Batch Size:** {metrics['batch_size']}  \n"
            
            if metrics['learning_rate']:
                details += f"**Learning Rate:** {metrics['learning_rate']}  \n"
            
            details += "\n"
        
        return details
    
    def _add_conclusions(self) -> str:
        """Add conclusions and analysis."""
        conclusions = """
## Conclusions and Analysis

### Architecture Comparison

1. **VDCNN**: Demonstrated strong performance with deep convolutional layers and K-max pooling
2. **DenseNet-BC**: Showed efficient parameter usage with dense connections and feature reuse

### Dataset Performance

- Both models successfully trained on character-level text classification tasks
- Performance varies across different datasets reflecting their inherent complexity

### Technical Implementation

- All experiments successfully completed with proper logging and metrics tracking
- Modular configuration system enabled seamless switching between models and datasets
- GPU acceleration utilized effectively for training

### Future Work

1. Extended training with more epochs for comprehensive performance evaluation
2. Hyperparameter optimization for both architectures
3. Additional datasets and cross-validation experiments
4. Performance profiling and optimization analysis

---

*This report was automatically generated by the ExperimentAnalyzer system.*
"""
        return conclusions


def main():
    """Main function to run the analysis and generate reports."""
    parser = argparse.ArgumentParser(description="Analyze experiment logs and generate reports.")
    parser.add_argument(
        '--reports_dir', 
        type=str, 
        default=str(project_root / "REPORTS"), # Use dynamic project root for default uppercase REPORTS
        help="Path to the reports directory. Defaults to '{project_root}/REPORTS'."
    )
    args = parser.parse_args()

    print("🚀 Starting Experiment Analysis...")
    analyzer = ExperimentAnalyzer(reports_dir=args.reports_dir)
    
    print("1. Discovering completed experiments...")
    experiments = analyzer.discover_experiments()
    if not experiments:
        print("No completed experiments found!")
        return
        
    print(f"✅ Found {len(experiments)} completed experiments.")
    
    print("2. Generating performance plots...")
    plots = analyzer.generate_performance_plots()
    print(f"✅ Plots generated in: {plots['model_comparison'].parent}")

    print("3. Generating final report...")
    analyzer.generate_report("experiment_analysis.md")
    
    print("\n🎉 Analysis complete!")


if __name__ == '__main__':
    main() 