import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from main import data_preprocess
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pickle
import json
from datetime import datetime

# Import the models
from dann_bp_model import BPEstimationModel, DANNModel

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

def run_baseline_experiment(data, subject_id, training_minutes=3):
    """
    Run baseline experiment with the MTL model.
    
    Args:
        data: Dictionary containing processed data for all subjects
        subject_id: ID of the subject to test
        training_minutes: Number of minutes of training data to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nRunning baseline experiment for Subject {subject_id} with {training_minutes} minutes of training data")
    
    # Get data for the subject
    subject_data = data[subject_id]
    
    # Calculate number of samples for X minutes (assuming 60 BPM average)
    samples_per_minute = 60
    n_samples = training_minutes * samples_per_minute
    
    # Limit training data
    X_train = subject_data['train']['X'][:n_samples]
    y_train = subject_data['train']['y'][:n_samples]
    
    # Get validation and test data
    X_val = subject_data['val']['X']
    y_val = subject_data['val']['y']
    X_test = subject_data['test']['X']
    y_test = subject_data['test']['y']
    
    print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples, testing with {len(X_test)} samples")
    
    # Create and train the model
    model = BPEstimationModel(input_shape=(X_train.shape[1],))
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    
    print(f"DBP RMSE: {metrics['dbp_rmse']:.4f} mmHg, SBP RMSE: {metrics['sbp_rmse']:.4f} mmHg")
    print(f"DBP corr: {metrics['dbp_corr']:.4f}, SBP corr: {metrics['sbp_corr']:.4f}")
    print(f"DBP within 10 mmHg: {metrics['dbp_within_10']:.2f}%, SBP within 10 mmHg: {metrics['sbp_within_10']:.2f}%")
    
    return metrics

def run_pretrained_experiment(data, subject_id, source_subject_id, training_minutes=3):
    """
    Run experiment with a pretrained model from another subject.
    
    Args:
        data: Dictionary containing processed data for all subjects
        subject_id: ID of the subject to test
        source_subject_id: ID of the source subject for pretraining
        training_minutes: Number of minutes of training data to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nRunning pretrained experiment for Subject {subject_id} with source Subject {source_subject_id}")
    print(f"Using {training_minutes} minutes of training data")
    
    # Get data for the subjects
    target_data = data[subject_id]
    source_data = data[source_subject_id]
    
    # Calculate number of samples for X minutes
    samples_per_minute = 60
    n_samples = training_minutes * samples_per_minute
    
    # Get data for source subject (for pretraining)
    X_source_train = source_data['train']['X']
    y_source_train = source_data['train']['y']
    X_source_val = source_data['val']['X']
    y_source_val = source_data['val']['y']
    
    # Get limited data for target subject (for fine-tuning)
    X_target_train = target_data['train']['X'][:n_samples]
    y_target_train = target_data['train']['y'][:n_samples]
    X_target_val = target_data['val']['X']
    y_target_val = target_data['val']['y']
    X_target_test = target_data['test']['X']
    y_target_test = target_data['test']['y']
    
    print(f"Pretraining with {len(X_source_train)} samples from source subject")
    print(f"Fine-tuning with {len(X_target_train)} samples from target subject")
    
    # Create and pretrain the model on source subject
    model = BPEstimationModel(input_shape=(X_source_train.shape[1],))
    history = model.train(X_source_train, y_source_train, X_source_val, y_source_val, 
                         epochs=50, batch_size=32, verbose=0)
    
    # Fine-tune the model on target subject
    history = model.train(X_target_train, y_target_train, X_target_val, y_target_val, 
                         epochs=30, batch_size=32, verbose=0)
    
    # Evaluate on test set
    metrics = model.evaluate(X_target_test, y_target_test)
    
    print(f"DBP RMSE: {metrics['dbp_rmse']:.4f} mmHg, SBP RMSE: {metrics['sbp_rmse']:.4f} mmHg")
    print(f"DBP corr: {metrics['dbp_corr']:.4f}, SBP corr: {metrics['sbp_corr']:.4f}")
    print(f"DBP within 10 mmHg: {metrics['dbp_within_10']:.2f}%, SBP within 10 mmHg: {metrics['sbp_within_10']:.2f}%")
    
    return metrics

def run_dann_experiment(data, subject_id, source_subject_ids=None, training_minutes=3, 
                        num_runs=10):
    """
    Run DANN experiment for a subject with reduced training data.
    
    Args:
        data: Dictionary containing processed data for all subjects
        subject_id: ID of the subject to test
        source_subject_ids: List of source subject IDs (if None, random selection)
        training_minutes: Number of minutes of training data to use
        num_runs: Number of runs with different source subjects
        
    Returns:
        Dictionary of evaluation metrics averaged over runs
    """
    print(f"\nRunning DANN experiment for Subject {subject_id} with {training_minutes} minutes of training data")
    
    # Get all subject IDs except the target subject
    all_subject_ids = list(data.keys())
    available_source_ids = [id for id in all_subject_ids if id != subject_id]
    
    # Get test data for the target subject
    X_test = data[subject_id]['test']['X']
    y_test = data[subject_id]['test']['y']
    
    # Run multiple times with different source subjects
    all_metrics = []
    
    for run in range(num_runs):
        # Select source subjects if not provided
        if source_subject_ids is None:
            run_source_ids = random.sample(available_source_ids, 2)
        else:
            run_source_ids = source_subject_ids
            
        print(f"\nRun {run+1}/{num_runs} - Using source subjects {run_source_ids}")
        
        # Create and train DANN model
        dann_model = DANNModel(input_shape=(data[subject_id]['train']['X'].shape[1],), 
                              num_domains=3)  # 3 domains: target, source1, source2
        
        # Train the model
        history = dann_model.train(data, subject_id, run_source_ids, 
                                  training_minutes=training_minutes, 
                                  epochs=100, batch_size=32)
        
        # Evaluate the model
        metrics = dann_model.evaluate(X_test, y_test)
        all_metrics.append(metrics)
        
        print(f"DBP RMSE: {metrics['dbp_rmse']:.4f} mmHg, SBP RMSE: {metrics['sbp_rmse']:.4f} mmHg")
        print(f"DBP corr: {metrics['dbp_corr']:.4f}, SBP corr: {metrics['sbp_corr']:.4f}")
        print(f"DBP within 10 mmHg: {metrics['dbp_within_10']:.2f}%, SBP within 10 mmHg: {metrics['sbp_within_10']:.2f}%")
        
        # Create Bland-Altman plot
        plt.figure(figsize=(12, 5))
        fig = dann_model.plot_bland_altman(X_test, y_test, 
                                         title=f'Subject {subject_id} - {training_minutes} min training - Run {run+1}')
        plt.savefig(f'results/subject_{subject_id}_dann_{training_minutes}min_run{run+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    print("\nAverage metrics over all runs:")
    print(f"DBP RMSE: {avg_metrics['dbp_rmse']['mean']:.2f} ± {avg_metrics['dbp_rmse']['std']:.2f} mmHg")
    print(f"SBP RMSE: {avg_metrics['sbp_rmse']['mean']:.2f} ± {avg_metrics['sbp_rmse']['std']:.2f} mmHg")
    print(f"DBP corr: {avg_metrics['dbp_corr']['mean']:.2f} ± {avg_metrics['dbp_corr']['std']:.2f}")
    print(f"SBP corr: {avg_metrics['sbp_corr']['mean']:.2f} ± {avg_metrics['sbp_corr']['std']:.2f}")
    
    return avg_metrics

def run_all_experiments(data, training_minutes=4):
    """
    Run experiments for all subjects with the specified training minutes.
    
    Args:
        data: Dictionary containing processed data for all subjects
        training_minutes: Number of minutes of training data to use
        
    Returns:
        Dictionary of results
    """
    results = {
        'baseline': {},
        'pretrained': {},
        'dann': {}
    }
    
    # Get all subject IDs
    all_subject_ids = list(data.keys())
    
    # Run experiments for each subject
    for subject_id in all_subject_ids:
        print(f"\n{'='*80}")
        print(f"Subject {subject_id} Experiments with {training_minutes} minutes of training data")
        print(f"{'='*80}")
        
        # Baseline experiment
        results['baseline'][subject_id] = run_baseline_experiment(data, subject_id, training_minutes)
        
        # Pretrained experiments (try all other subjects as source)
        pretrained_results = []
        for source_id in [id for id in all_subject_ids if id != subject_id]:
            metrics = run_pretrained_experiment(data, subject_id, source_id, training_minutes)
            pretrained_results.append(metrics)
        
        # Average pretrained results
        avg_pretrained = {}
        for key in pretrained_results[0].keys():
            values = [r[key] for r in pretrained_results]
            avg_pretrained[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        results['pretrained'][subject_id] = avg_pretrained
        
        # DANN experiment
        results['dann'][subject_id] = run_dann_experiment(data, subject_id, training_minutes=training_minutes)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/experiment_results_{training_minutes}min_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

def summarize_results(results, training_minutes):
    """
    Summarize and compare results across methods.
    
    Args:
        results: Dictionary of results from run_all_experiments
        training_minutes: Number of minutes of training data used
    """
    summary = {
        'dbp_rmse': {'baseline': [], 'pretrained': [], 'dann': []},
        'sbp_rmse': {'baseline': [], 'pretrained': [], 'dann': []},
        'dbp_corr': {'baseline': [], 'pretrained': [], 'dann': []},
        'sbp_corr': {'baseline': [], 'pretrained': [], 'dann': []}
    }
    
    # Collect metrics for all subjects
    for subject_id in results['baseline'].keys():
        # Baseline
        summary['dbp_rmse']['baseline'].append(results['baseline'][subject_id]['dbp_rmse'])
        summary['sbp_rmse']['baseline'].append(results['baseline'][subject_id]['sbp_rmse'])
        summary['dbp_corr']['baseline'].append(results['baseline'][subject_id]['dbp_corr'])
        summary['sbp_corr']['baseline'].append(results['baseline'][subject_id]['sbp_corr'])
        
        # Pretrained
        summary['dbp_rmse']['pretrained'].append(results['pretrained'][subject_id]['dbp_rmse']['mean'])
        summary['sbp_rmse']['pretrained'].append(results['pretrained'][subject_id]['sbp_rmse']['mean'])
        summary['dbp_corr']['pretrained'].append(results['pretrained'][subject_id]['dbp_corr']['mean'])
        summary['sbp_corr']['pretrained'].append(results['pretrained'][subject_id]['sbp_corr']['mean'])
        
        # DANN
        summary['dbp_rmse']['dann'].append(results['dann'][subject_id]['dbp_rmse']['mean'])
        summary['sbp_rmse']['dann'].append(results['dann'][subject_id]['sbp_rmse']['mean'])
        summary['dbp_corr']['dann'].append(results['dann'][subject_id]['dbp_corr']['mean'])
        summary['sbp_corr']['dann'].append(results['dann'][subject_id]['sbp_corr']['mean'])
    
    # Calculate average metrics
    avg_summary = {}
    for metric, methods in summary.items():
        avg_summary[metric] = {}
        for method, values in methods.items():
            avg_summary[metric][method] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary of Results with {training_minutes} minutes of training data")
    print(f"{'='*80}")
    
    print("\nDBP RMSE (mmHg):")
    print(f"Baseline: {avg_summary['dbp_rmse']['baseline']['mean']:.2f} ± {avg_summary['dbp_rmse']['baseline']['std']:.2f}")
    print(f"Pretrained: {avg_summary['dbp_rmse']['pretrained']['mean']:.2f} ± {avg_summary['dbp_rmse']['pretrained']['std']:.2f}")
    print(f"DANN: {avg_summary['dbp_rmse']['dann']['mean']:.2f} ± {avg_summary['dbp_rmse']['dann']['std']:.2f}")
    
    print("\nSBP RMSE (mmHg):")
    print(f"Baseline: {avg_summary['sbp_rmse']['baseline']['mean']:.2f} ± {avg_summary['sbp_rmse']['baseline']['std']:.2f}")
    print(f"Pretrained: {avg_summary['sbp_rmse']['pretrained']['mean']:.2f} ± {avg_summary['sbp_rmse']['pretrained']['std']:.2f}")
    print(f"DANN: {avg_summary['sbp_rmse']['dann']['mean']:.2f} ± {avg_summary['sbp_rmse']['dann']['std']:.2f}")
    
    print("\nDBP Correlation:")
    print(f"Baseline: {avg_summary['dbp_corr']['baseline']['mean']:.2f} ± {avg_summary['dbp_corr']['baseline']['std']:.2f}")
    print(f"Pretrained: {avg_summary['dbp_corr']['pretrained']['mean']:.2f} ± {avg_summary['dbp_corr']['pretrained']['std']:.2f}")
    print(f"DANN: {avg_summary['dbp_corr']['dann']['mean']:.2f} ± {avg_summary['dbp_corr']['dann']['std']:.2f}")
    
    print("\nSBP Correlation:")
    print(f"Baseline: {avg_summary['sbp_corr']['baseline']['mean']:.2f} ± {avg_summary['sbp_corr']['baseline']['std']:.2f}")
    print(f"Pretrained: {avg_summary['sbp_corr']['pretrained']['mean']:.2f} ± {avg_summary['sbp_corr']['pretrained']['std']:.2f}")
    print(f"DANN: {avg_summary['sbp_corr']['dann']['mean']:.2f} ± {avg_summary['sbp_corr']['dann']['std']:.2f}")
    
    # Create bar chart for RMSE comparison
    plt.figure(figsize=(12, 6))
    
    methods = ['Baseline', 'Pretrained', 'DANN']
    x = np.arange(len(methods))
    width = 0.35
    
    dbp_means = [avg_summary['dbp_rmse'][m]['mean'] for m in ['baseline', 'pretrained', 'dann']]
    dbp_stds = [avg_summary['dbp_rmse'][m]['std'] for m in ['baseline', 'pretrained', 'dann']]
    
    sbp_means = [avg_summary['sbp_rmse'][m]['mean'] for m in ['baseline', 'pretrained', 'dann']]
    sbp_stds = [avg_summary['sbp_rmse'][m]['std'] for m in ['baseline', 'pretrained', 'dann']]
    
    plt.bar(x - width/2, dbp_means, width, label='DBP', yerr=dbp_stds)
    plt.bar(x + width/2, sbp_means, width, label='SBP', yerr=sbp_stds)
    
    plt.ylabel('RMSE (mmHg)')
    plt.title(f'Blood Pressure Estimation RMSE ({training_minutes} min training)')
    plt.xticks(x, methods)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/rmse_comparison_{training_minutes}min.png', dpi=300)
    plt.show()
    
    return avg_summary

def main():
    # Load or preprocess data
    try:
        print("Trying to load preprocessed data...")
        with open('data/processed_data.pkl', 'rb') as f:
            processed_data = pickle.load(f)
        print("Loaded preprocessed data.")
    except (FileNotFoundError, EOFError):
        print("Preprocessing data...")
        processed_data = data_preprocess()
        # Save processed data
        os.makedirs('data', exist_ok=True)
        with open('data/processed_data.pkl', 'wb') as f:
            pickle.dump(processed_data, f)
        print("Data preprocessing complete.")
    
    # Run experiments with different amounts of training data
    for minutes in [3, 4, 5]:
        print(f"\n{'#'*100}")
        print(f"Running experiments with {minutes} minutes of training data")
        print(f"{'#'*100}")
        
        results = run_all_experiments(processed_data, training_minutes=minutes)
        summary = summarize_results(results, minutes)

if __name__ == "__main__":
    main()
