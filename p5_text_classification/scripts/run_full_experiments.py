#!/usr/bin/env python3
"""
Full Experiments Runner with Enhanced Debugging
Runs all four model-dataset combinations with detailed logging and debugging information.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_debug(message, level="INFO"):
    """Print debug message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    sys.stdout.flush()  # Force output immediately

def run_experiment(model, dataset, max_samples=None, epochs=10):
    """
    Run a single experiment with detailed debugging
    """
    print_debug(f"🚀 Starting experiment: {model.upper()} + {dataset.upper()}")
    
    # Build command
    train_script = PROJECT_ROOT / "p5_text_classification" / "src" / "train.py"
    print_debug(f"📄 Using train script: {train_script}")
    
    if not train_script.exists():
        print_debug(f"❌ ERROR: Train script not found at {train_script}", "ERROR")
        return False
    
    cmd = [
        sys.executable,
        str(train_script),
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs)
    ]
    
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    
    print_debug(f"📋 Command: {' '.join(cmd)}")
    
    # Set environment
        env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    print_debug(f"🔧 PYTHONPATH set to: {PROJECT_ROOT}")

    # Start experiment
        start_time = time.time()
    print_debug(f"⏰ Experiment started at: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        print_debug("🔄 Starting subprocess...")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        print_debug(f"✅ Process started with PID: {process.pid}")
        
        # Monitor process output in real-time
        line_count = 0
        last_output_time = time.time()
        
        while True:
            output = process.stdout.readline()
            current_time = time.time()
            
            if output == '' and process.poll() is not None:
                print_debug("📄 Process finished, no more output")
                break
                
            if output:
                line_count += 1
                last_output_time = current_time
                # Print every line with timestamp
                print_debug(f"[OUTPUT {line_count:04d}] {output.strip()}")
            else:
                # Check if process is hanging
                if current_time - last_output_time > 30:  # 30 seconds without output
                    print_debug(f"⚠️  WARNING: No output for {current_time - last_output_time:.1f} seconds", "WARN")
                    print_debug(f"📊 Process status: {'running' if process.poll() is None else 'finished'}")
                    last_output_time = current_time  # Reset timer
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
        
        # Wait for process to complete
        print_debug("⏳ Waiting for process to complete...")
        return_code = process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print_debug(f"✅ Process completed with return code: {return_code}")
        print_debug(f"⏰ Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        if return_code == 0:
            print_debug(f"🎉 SUCCESS: {model.upper()} + {dataset.upper()} completed successfully!")
            return True
        else:
            print_debug(f"❌ FAILED: {model.upper()} + {dataset.upper()} failed with code {return_code}", "ERROR")
            return False
            
    except Exception as e:
        print_debug(f"💥 EXCEPTION during {model}+{dataset}: {str(e)}", "ERROR")
        return False

def check_environment():
    """Check if environment is properly set up"""
    print_debug("🔍 Checking environment...")
    
    # Check Python version
    print_debug(f"🐍 Python version: {sys.version}")
    
    # Check if we're in the right directory
    print_debug(f"📁 Current working directory: {os.getcwd()}")
    print_debug(f"📁 Project root: {PROJECT_ROOT}")
    
    # Check if train.py exists
    train_script = PROJECT_ROOT / "p5_text_classification" / "src" / "train.py"
    if train_script.exists():
        print_debug(f"✅ Train script found: {train_script}")
    else:
        print_debug(f"❌ Train script NOT found: {train_script}", "ERROR")
        return False
    
    # Check if data directories exist
    data_dir = PROJECT_ROOT / "p5_text_classification" / "data"
    for dataset in ["yelp", "yahoo"]:
        dataset_dir = data_dir / dataset
        if dataset_dir.exists():
            print_debug(f"✅ Dataset directory found: {dataset_dir}")
        else:
            print_debug(f"❌ Dataset directory NOT found: {dataset_dir}", "ERROR")
            return False
    
    # Check REPORTS directory
    reports_dir = PROJECT_ROOT / "REPORTS"
    if not reports_dir.exists():
        print_debug(f"📁 Creating REPORTS directory: {reports_dir}")
        reports_dir.mkdir(exist_ok=True)
    else:
        print_debug(f"✅ REPORTS directory exists: {reports_dir}")
    
    print_debug("✅ Environment check completed successfully!")
    return True

def main():
    """Main function to run all experiments"""
    print_debug("="*80)
    print_debug("🚀 FULL EXPERIMENTS RUNNER STARTED")
    print_debug("="*80)
    
    start_time = time.time()
    
    # Check environment first
    if not check_environment():
        print_debug("❌ Environment check failed, aborting!", "ERROR")
        return 1
    
    # Define experiments
    experiments = [
        ("vdcnn", "yelp"),
        ("vdcnn", "yahoo"),
        ("densenet", "yelp"),
        ("densenet", "yahoo")
    ]
    
    print_debug(f"📋 Planning to run {len(experiments)} experiments:")
    for i, (model, dataset) in enumerate(experiments, 1):
        print_debug(f"  {i}. {model.upper()} + {dataset.upper()}")
    
    # Ask user for configuration
    print_debug("🔧 Configuration options:")
    print_debug("  - For quick test: Use --test flag (500 samples, 2 epochs)")
    print_debug("  - For full experiments: No flags (full dataset, 10 epochs)")
    
    # Check command line arguments
    test_mode = "--test" in sys.argv
    if test_mode:
        print_debug("🧪 TEST MODE: Running with limited samples and epochs")
        max_samples = 500
        epochs = 2
    else:
        print_debug("🎯 FULL MODE: Running with complete dataset and 10 epochs")
        max_samples = None
        epochs = 10
    
    # Run experiments
    results = {}
    successful = 0
    failed = 0
    
    for i, (model, dataset) in enumerate(experiments, 1):
        print_debug("="*60)
        print_debug(f"🎯 EXPERIMENT {i}/{len(experiments)}: {model.upper()} + {dataset.upper()}")
        print_debug("="*60)
        
        success = run_experiment(model, dataset, max_samples, epochs)
        results[f"{model}_{dataset}"] = success
        
        if success:
            successful += 1
            print_debug(f"✅ Experiment {i} completed successfully!")
        else:
            failed += 1
            print_debug(f"❌ Experiment {i} failed!", "ERROR")
        
        print_debug(f"📊 Progress: {i}/{len(experiments)} completed ({successful} successful, {failed} failed)")
        
        if i < len(experiments):
            print_debug("⏸️  Waiting 5 seconds before next experiment...")
            time.sleep(5)
    
    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print_debug("="*80)
    print_debug("📊 FINAL RESULTS SUMMARY")
    print_debug("="*80)
    print_debug(f"⏰ Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print_debug(f"✅ Successful experiments: {successful}/{len(experiments)}")
    print_debug(f"❌ Failed experiments: {failed}/{len(experiments)}")
    
    print_debug("\n📋 Detailed results:")
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print_debug(f"  {exp_name}: {status}")
    
    if failed == 0:
        print_debug("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print_debug(f"⚠️  {failed} experiments failed. Check logs above for details.", "WARN")
        return 1

if __name__ == "__main__":
    print_debug("🏁 Script started")
    try:
        exit_code = main()
        print_debug(f"🏁 Script finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_debug("⚠️  Script interrupted by user (Ctrl+C)", "WARN")
        sys.exit(130)
    except Exception as e:
        print_debug(f"💥 Unexpected error: {str(e)}", "ERROR")
        import traceback
        print_debug(f"📋 Traceback:\n{traceback.format_exc()}", "ERROR")
        sys.exit(1) 