#!/usr/bin/env python3
"""
Quick Test Runner for All Model-Dataset Combinations

This script automatically runs quick tests (small sample size, few epochs) 
for all 4 experiment combinations to verify functionality and performance.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# --- Start of Path Hack ---
# Ensures the script can find project modules, regardless of execution location.
# This section is no longer strictly necessary if PYTHONPATH is set correctly,
# but it provides an extra layer of robustness.
def find_project_root_for_path() -> Path:
    """Dynamically finds the project's root directory for sys.path."""
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / 'README.md').exists() and (parent / 'p5_text_classification').is_dir():
            return parent
    raise FileNotFoundError("Could not find project root. Make sure 'README.md' and 'p5_text_classification' exist.")

try:
    PROJECT_ROOT_FOR_PATH = find_project_root_for_path()
    if str(PROJECT_ROOT_FOR_PATH) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_FOR_PATH))
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Could not configure sys.path. {e}")
    sys.exit(1)
# --- End of Path Hack ---

# Now that the path is set, we can import project modules
from p5_text_classification.src.utils.constants import PROJECT_ROOT, REPORTS_DIR


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class QuickTestRunner:
    """
    Runner for quick validation tests across all model-dataset combinations.
    
    Responsibilities:
    - Execute all 4 core experiment combinations
    - Monitor test execution and capture results
    - Generate summary reports
    - Validate system functionality
    """
    
    def __init__(self, 
                 max_samples: int = 100, 
                 epochs: int = 2):
        """
        Initialize the quick test runner.
        
        Args:
            max_samples: Maximum samples for quick testing
            epochs: Number of epochs for quick testing
        """
        self.max_samples = max_samples
        self.epochs = epochs
        
        # Define all experiment combinations
        self.experiments = [
            {"model": "vdcnn", "dataset": "yelp", "description": "VDCNN on Yelp Reviews"},
            {"model": "densenet", "dataset": "yelp", "description": "DenseNet on Yelp Reviews"},
            {"model": "vdcnn", "dataset": "yahoo", "description": "VDCNN on Yahoo Answers"},
            {"model": "densenet", "dataset": "yahoo", "description": "DenseNet on Yahoo Answers"}
        ]
        
        self.results: List[Dict] = []
        
    def run_single_experiment(self, experiment: Dict) -> Dict:
        """
        Run a single experiment and capture results.
        
        Args:
            experiment: Experiment configuration dictionary
            
        Returns:
            Dictionary containing experiment results
        """
        model = experiment["model"]
        dataset = experiment["dataset"]
        description = experiment["description"]
        
        logger.info(f"🚀 Starting: {description}")
        logger.info(f"   Parameters: --model {model} --dataset {dataset} --epochs {self.epochs} --max_samples {self.max_samples}")
        
        # Construct command using python -m for robust execution
        cmd = [
            sys.executable, "-m", "p5_text_classification.main",
            "--model", model,
            "--dataset", dataset,
        ]
        
        # Add optional arguments only if they have a value
        if self.epochs is not None:
            cmd.extend(["--epochs", str(self.epochs)])
        if self.max_samples is not None:
            cmd.extend(["--max_samples", str(self.max_samples)])
        
        # Record start time
        start_time = time.time()
        
        # --- The Definitive PYTHONPATH Fix ---
        # Create a copy of the current environment and explicitly set PYTHONPATH
        # to the project root. This guarantees that the subprocess can find
        # the 'p5_text_classification' module.
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        try:
            # Run the experiment, ensuring the CWD is the project root
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )
            
            # Record end time
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse results
            success = result.returncode == 0
            
            # Extract key metrics from output
            final_accuracy = self._extract_final_accuracy(result.stdout)
            model_parameters = self._extract_model_parameters(result.stdout)
            
            experiment_result = {
                "experiment": experiment,
                "success": success,
                "duration": duration,
                "final_accuracy": final_accuracy,
                "model_parameters": model_parameters,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            if success:
                logger.info(f"✅ Completed: {description}")
                logger.info(f"   Duration: {duration:.1f}s")
                if final_accuracy:
                    logger.info(f"   Final Accuracy: {final_accuracy:.1f}%")
                if model_parameters:
                    logger.info(f"   Model Parameters: {model_parameters:,}")
            else:
                logger.error(f"❌ Failed: {description}")
                logger.error(f"   Return Code: {result.returncode}")
                if result.stderr:
                    logger.error(f"   Error: {result.stderr[:200]}...")
            
            return experiment_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout: {description} (exceeded 5 minutes)")
            return {
                "experiment": experiment,
                "success": False,
                "duration": 300,
                "final_accuracy": None,
                "model_parameters": None,
                "stdout": "",
                "stderr": "Timeout expired",
                "return_code": -1
            }
            
        except Exception as e:
            logger.error(f"💥 Exception in {description}: {str(e)}")
            return {
                "experiment": experiment,
                "success": False,
                "duration": 0,
                "final_accuracy": None,
                "model_parameters": None,
                "stdout": "",
                "stderr": str(e),
                "return_code": -2
            }
    
    def _extract_final_accuracy(self, output: str) -> float:
        """Extract final validation accuracy from output."""
        import re
        match = re.search(r'Final validation accuracy: ([\d.]+)%', output)
        return float(match.group(1)) if match else None
    
    def _extract_model_parameters(self, output: str) -> int:
        """Extract model parameter count from output."""
        import re
        match = re.search(r"Model '\w+' created with ([\d,]+) parameters", output)
        return int(match.group(1).replace(',', '')) if match else None
    
    def run_all_experiments(self) -> List[Dict]:
        """
        Run all experiment combinations.
        
        Returns:
            List of experiment results
        """
        logger.info("=" * 80)
        logger.info("🧪 STARTING QUICK TEST SUITE")
        logger.info("=" * 80)
        logger.info(f"📊 Total Experiments: {len(self.experiments)}")
        logger.info(f"⚙️  Test Parameters: {self.max_samples} samples, {self.epochs} epochs")
        logger.info(f"📂 Working Directory: {PROJECT_ROOT}")
        logger.info("=" * 80)
        
        self.results = []
        
        for i, experiment in enumerate(self.experiments, 1):
            logger.info(f"\n[{i}/{len(self.experiments)}] " + "=" * 50)
            
            result = self.run_single_experiment(experiment)
            self.results.append(result)
            
            # Small delay between experiments
            if i < len(self.experiments):
                time.sleep(2)
        
        return self.results
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of all test results.
        
        Returns:
            Formatted summary report string
        """
        if not self.results:
            return "No test results available."
        
        # Calculate summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - successful_tests
        total_duration = sum(r["duration"] for r in self.results)
        
        # Create report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Quick Test Suite Report

**Generated:** {timestamp}
**Test Configuration:** {self.max_samples} samples, {self.epochs} epochs

## Summary

- **Total Tests:** {total_tests}
- **Successful:** {successful_tests} ✅
- **Failed:** {failed_tests} ❌
- **Success Rate:** {(successful_tests/total_tests)*100:.1f}%
- **Total Duration:** {total_duration:.1f} seconds

## Individual Test Results

| # | Model | Dataset | Status | Duration | Accuracy | Parameters |
|---|-------|---------|--------|----------|----------|------------|"""

        for i, result in enumerate(self.results, 1):
            exp = result["experiment"]
            status = "✅ Pass" if result["success"] else "❌ Fail"
            duration = f"{result['duration']:.1f}s"
            accuracy = f"{result['final_accuracy']:.1f}%" if result['final_accuracy'] else "N/A"
            params = f"{result['model_parameters']:,}" if result['model_parameters'] else "N/A"
            
            report += f"\n| {i} | {exp['model'].upper()} | {exp['dataset'].upper()} | {status} | {duration} | {accuracy} | {params} |"
        
            # Add failure details
            if not result["success"]:
                report += f"\n| |<td colspan='6'>**Error:**<br><pre><code>{result['stderr'] or 'No stderr output.'}</code></pre></td>| "

        # Add failure details if any
        failures = [r for r in self.results if not r["success"]]
        if failures:
            report += "\n\n## Failure Details\n"
            for i, failure in enumerate(failures, 1):
                exp = failure["experiment"]
                report += f"\n### {i}. {exp['model'].upper()} on {exp['dataset'].upper()}\n"
                report += f"- **Return Code:** {failure['return_code']}\n"
                report += f"- **Error:** {failure['stderr'][:200]}...\n"
        
        # Add recommendations
        report += self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on test results."""
        successful_tests = sum(1 for r in self.results if r["success"])
        total_tests = len(self.results)
        
        recommendations = "\n\n## Recommendations\n"
        
        if successful_tests == total_tests:
            recommendations += """
🎉 **All tests passed successfully!**

- The system is ready for full-scale experiments
- Consider running complete training with more epochs
- All model-dataset combinations are functional
- Log analysis and reporting tools can be utilized
"""
        elif successful_tests > 0:
            recommendations += f"""
⚠️ **{successful_tests}/{total_tests} tests passed**

- Investigate failed experiments for configuration issues
- Check data paths and model configurations
- Verify system dependencies and GPU availability
- Review error logs for specific failure causes
"""
        else:
            recommendations += """
🚨 **All tests failed**

- Check Python environment and package installations
- Verify data files are present and accessible
- Confirm GPU drivers and CUDA installation
- Review project configuration and paths
"""
        
        return recommendations
    
    def save_report(self, filename: str = None) -> Path:
        """
        Save the summary report to a file in the project's main REPORTS directory.
        """
        report_str = self.generate_summary_report()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quick_test_report_{timestamp}.md"
            
        # Use the SSOT for the reports directory from constants
        report_path = REPORTS_DIR / filename
        
        try:
            report_path.write_text(report_str, encoding='utf-8')
            logger.info(f"📄 Report saved to: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Could not save report to {report_path}: {e}")
            return None


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run quick functionality tests for all experiments.")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of samples to use for testing.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for testing.")
    args = parser.parse_args()

    runner = QuickTestRunner(
        max_samples=args.max_samples,
        epochs=args.epochs
    )
    
    results = runner.run_all_experiments()
    runner.save_report()
    
    # Exit with a non-zero code if any test failed
    if any(not r['success'] for r in results):
        sys.exit(1)

if __name__ == "__main__":
    main() 