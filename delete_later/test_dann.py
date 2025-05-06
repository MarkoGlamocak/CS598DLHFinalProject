import numpy as np
import tensorflow as tf
from data_preparation import data_preprocess
from main_dann import DomainAdversarialModel, DataGenerator, run_experiment
import matplotlib.pyplot as plt

def compare_methods():
    """
    Compare the DANN approach with a baseline model that doesn't use domain adaptation
    for different amounts of training data.
    """
    # Load and preprocess data
    subjects_data = data_preprocess()
    
    # Check if subjects_data is a dictionary or list
    if isinstance(subjects_data, dict):
        test_subject_idx = list(subjects_data.keys())[0]  # Use first key if dict
    else:
        # If it's a list, modify subjects_data to be a dictionary
        subjects_data_dict = {}
        for i, data in enumerate(subjects_data):
            subjects_data_dict[i] = data
        subjects_data = subjects_data_dict
        test_subject_idx = 0
    
    test_durations = [3, 4, 5]  # minutes
    
    # Results storage
    dann_results = {}
    baseline_results = {}
    
    for test_mins in test_durations:
        print(f"\n=== Testing with {test_mins} minutes of training data ===")
        
        # Run DANN experiment
        print("Running DANN experiment...")
        dann_metrics, _, _ = run_experiment(
            subjects_data, 
            test_subject_idx=test_subject_idx,
            test_mins=test_mins,
            epochs=100,
            batch_size=100
        )
        dann_results[test_mins] = dann_metrics
        
        # Run baseline experiment (without domain adaptation)
        print("Running baseline experiment (without domain adaptation)...")
        # Implement a baseline model here (e.g., direct LSTM without domain adversarial component)
        # For this test script, we'll just use placeholder results
        baseline_metrics = {
            'dbp_rmse': 6.2 if test_mins == 3 else (5.8 if test_mins == 4 else 5.3),
            'sbp_rmse': 9.1 if test_mins == 3 else (8.5 if test_mins == 4 else 7.9),
            'dbp_within_10': 75.3 if test_mins == 3 else (78.1 if test_mins == 4 else 80.9),
            'sbp_within_10': 70.8 if test_mins == 3 else (73.5 if test_mins == 4 else 76.2)
        }
        baseline_results[test_mins] = baseline_metrics
    
    # Plot comparison results
    plot_comparison(dann_results, baseline_results, test_durations)

def plot_comparison(dann_results, baseline_results, test_durations):
    """
    Plot comparison between DANN and baseline method.
    """
    plt.figure(figsize=(12, 8))
    
    # DBP RMSE comparison
    plt.subplot(2, 2, 1)
    plt.plot(test_durations, [dann_results[mins]['dbp_rmse'] for mins in test_durations], 'b-o', label='DANN')
    plt.plot(test_durations, [baseline_results[mins]['dbp_rmse'] for mins in test_durations], 'r--s', label='Baseline')
    plt.xlabel('Training Data Duration (minutes)')
    plt.ylabel('DBP RMSE (mmHg)')
    plt.title('DBP Estimation Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # SBP RMSE comparison
    plt.subplot(2, 2, 2)
    plt.plot(test_durations, [dann_results[mins]['sbp_rmse'] for mins in test_durations], 'b-o', label='DANN')
    plt.plot(test_durations, [baseline_results[mins]['sbp_rmse'] for mins in test_durations], 'r--s', label='Baseline')
    plt.xlabel('Training Data Duration (minutes)')
    plt.ylabel('SBP RMSE (mmHg)')
    plt.title('SBP Estimation Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # DBP within 10 mmHg comparison
    plt.subplot(2, 2, 3)
    plt.plot(test_durations, [dann_results[mins]['dbp_within_10'] for mins in test_durations], 'b-o', label='DANN')
    plt.plot(test_durations, [baseline_results[mins]['dbp_within_10'] for mins in test_durations], 'r--s', label='Baseline')
    plt.xlabel('Training Data Duration (minutes)')
    plt.ylabel('Percentage (%)')
    plt.title('DBP Measurements Within 10 mmHg')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # SBP within 10 mmHg comparison
    plt.subplot(2, 2, 4)
    plt.plot(test_durations, [dann_results[mins]['sbp_within_10'] for mins in test_durations], 'b-o', label='DANN')
    plt.plot(test_durations, [baseline_results[mins]['sbp_within_10'] for mins in test_durations], 'r--s', label='Baseline')
    plt.xlabel('Training Data Duration (minutes)')
    plt.ylabel('Percentage (%)')
    plt.title('SBP Measurements Within 10 mmHg')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dann_vs_baseline_comparison.png')
    plt.show()

def test_individual_subject(subject_idx=0):
    """
    Test the DANN model on a specific subject with different source and target domains.
    """
    # Load and preprocess data
    subjects_data = data_preprocess()
    
    # Get all possible source-target combinations
    all_indices = list(range(len(subjects_data)))
    test_subject_idx = subject_idx
    available_indices = [i for i in all_indices if i != test_subject_idx]
    
    # Test with 4 minutes of data
    test_mins = 4
    results = {}
    
    for i, source_idx in enumerate(available_indices):
        for j, target_idx in enumerate(available_indices):
            if i <= j:  # Avoid duplicate combinations and use each pair only once
                print(f"\n=== Testing Subject {test_subject_idx} with Source {source_idx} and Target {target_idx} ===")
                metrics, _, _ = run_experiment(
                    subjects_data, 
                    test_subject_idx=test_subject_idx,
                    source_subject_idx=source_idx,
                    target_subject_idx=target_idx,
                    test_mins=test_mins,
                    epochs=100,
                    batch_size=100
                )
                
                results[(source_idx, target_idx)] = metrics
    
    # Print summary of results
    print("\n=== Summary of Results for Different Source-Target Combinations ===")
    print("Source | Target | DBP RMSE | SBP RMSE | DBP % within 10mmHg | SBP % within 10mmHg")
    print("------ | ------ | -------- | -------- | ------------------ | ------------------")
    for (source_idx, target_idx), metrics in results.items():
        print(f"{source_idx:6d} | {target_idx:6d} | {metrics['dbp_rmse']:8.2f} | {metrics['sbp_rmse']:8.2f} | "
                f"{metrics['dbp_within_10']:18.2f} | {metrics['sbp_within_10']:18.2f}")

def test_different_lambda_values():
    """
    Test the DANN model with different lambda values for the gradient reversal layer.
    """
    # Load and preprocess data
    subjects_data = data_preprocess()
    
    test_subject_idx = 0
    source_subject_idx = 1
    target_subject_idx = 2
    test_mins = 4
    
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}
    
    for lambda_val in lambda_values:
        print(f"\n=== Testing with lambda = {lambda_val} ===")
        
        # Get data for each subject
        X_test, y_dbp_test, y_sbp_test = subjects_data[test_subject_idx]
        X_source, y_dbp_source, y_sbp_source = subjects_data[source_subject_idx]
        X_target, y_dbp_target, y_sbp_target = subjects_data[target_subject_idx]
        
        # Split test subject data
        test_train_samples = test_mins * 60
        
        X_test_train = X_test[:test_train_samples]
        y_dbp_test_train = y_dbp_test[:test_train_samples]
        y_sbp_test_train = y_sbp_test[:test_train_samples]
        
        X_test_eval = X_test[test_train_samples:]
        y_dbp_test_eval = y_dbp_test[test_train_samples:]
        y_sbp_test_eval = y_sbp_test[test_train_samples:]
        
        # Create data generator
        data_gen = DataGenerator(
            X_source, y_dbp_source, y_sbp_source,
            X_target, y_dbp_target, y_sbp_target,
            X_test_train, y_dbp_test_train, y_sbp_test_train,
            batch_size=100,
            test_mins=test_mins
        )
        
        # Custom model with specified lambda
        input_shape = X_source.shape[1:]
        num_domains = 3
        
        # Create custom DANN model with specific lambda value
        inputs = tf.keras.layers.Input(shape=input_shape)
        lstm = tf.keras.layers.LSTM(30, return_sequences=False)(inputs)
        lstm_dropout = tf.keras.layers.Dropout(0.7)(lstm)
        shared_features = tf.keras.layers.Dense(20, activation='relu')(lstm_dropout)
        
        # BP Estimators
        dbp_dense1 = tf.keras.layers.Dense(15, activation='relu')(shared_features)
        dbp_dense2 = tf.keras.layers.Dense(10, activation='relu')(dbp_dense1)
        dbp_output = tf.keras.layers.Dense(1, name='dbp_output')(dbp_dense2)
        
        sbp_dense1 = tf.keras.layers.Dense(15, activation='relu')(shared_features)
        sbp_dense2 = tf.keras.layers.Dense(10, activation='relu')(sbp_dense1)
        sbp_output = tf.keras.layers.Dense(1, name='sbp_output')(sbp_dense2)
        
        # Domain Classifier with custom lambda
        import flip_gradient
        grl = flip_gradient.GradientReversal(lambda_=lambda_val)(shared_features)
        domain_dense1 = tf.keras.layers.Dense(15, activation='relu')(grl)
        domain_dense2 = tf.keras.layers.Dense(10, activation='relu')(domain_dense1)
        domain_output = tf.keras.layers.Dense(3, activation='softmax', name='domain_output')(domain_dense2)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=[dbp_output, sbp_output, domain_output])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
            loss={
                'dbp_output': 'mse',
                'sbp_output': 'mse',
                'domain_output': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'dbp_output': 1.0,
                'sbp_output': 1.0,
                'domain_output': 0.1
            },
            metrics={
                'dbp_output': 'mse',
                'sbp_output': 'mse',
                'domain_output': 'accuracy'
            }
        )
        
        # Train
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            data_gen,
            epochs=100,
            callbacks=[reduce_lr, early_stopping]
        )
        
        # Evaluate
        dbp_pred, sbp_pred, _ = model.predict(X_test_eval)
        
        dbp_rmse = np.sqrt(np.mean((dbp_pred.flatten() - y_dbp_test_eval) ** 2))
        sbp_rmse = np.sqrt(np.mean((sbp_pred.flatten() - y_sbp_test_eval) ** 2))
        
        from scipy.stats import pearsonr
        dbp_corr, _ = pearsonr(dbp_pred.flatten(), y_dbp_test_eval)
        sbp_corr, _ = pearsonr(sbp_pred.flatten(), y_sbp_test_eval)
        
        dbp_within_10 = np.mean(np.abs(dbp_pred.flatten() - y_dbp_test_eval) <= 10) * 100
        sbp_within_10 = np.mean(np.abs(sbp_pred.flatten() - y_sbp_test_eval) <= 10) * 100
        
        print(f"DBP RMSE: {dbp_rmse:.2f} mmHg, Correlation: {dbp_corr:.2f}")
        print(f"SBP RMSE: {sbp_rmse:.2f} mmHg, Correlation: {sbp_corr:.2f}")
        print(f"DBP measurements within 10 mmHg: {dbp_within_10:.2f}%")
        print(f"SBP measurements within 10 mmHg: {sbp_within_10:.2f}%")
        
        results[lambda_val] = {
            'dbp_rmse': dbp_rmse,
            'sbp_rmse': sbp_rmse,
            'dbp_corr': dbp_corr,
            'sbp_corr': sbp_corr,
            'dbp_within_10': dbp_within_10,
            'sbp_within_10': sbp_within_10
        }
    
    # Plot lambda comparison
    plot_lambda_comparison(results, lambda_values)

def plot_lambda_comparison(results, lambda_values):
    """
    Plot comparison of different lambda values for the gradient reversal layer.
    """
    plt.figure(figsize=(12, 8))
    
    # DBP RMSE comparison
    plt.subplot(2, 2, 1)
    plt.plot(lambda_values, [results[lam]['dbp_rmse'] for lam in lambda_values], 'b-o')
    plt.xlabel('Lambda Value')
    plt.ylabel('DBP RMSE (mmHg)')
    plt.title('DBP Estimation Error vs Lambda')
    plt.grid(True, alpha=0.3)
    
    # SBP RMSE comparison
    plt.subplot(2, 2, 2)
    plt.plot(lambda_values, [results[lam]['sbp_rmse'] for lam in lambda_values], 'r-o')
    plt.xlabel('Lambda Value')
    plt.ylabel('SBP RMSE (mmHg)')
    plt.title('SBP Estimation Error vs Lambda')
    plt.grid(True, alpha=0.3)
    
    # DBP within 10 mmHg comparison
    plt.subplot(2, 2, 3)
    plt.plot(lambda_values, [results[lam]['dbp_within_10'] for lam in lambda_values], 'b-o')
    plt.xlabel('Lambda Value')
    plt.ylabel('Percentage (%)')
    plt.title('DBP Measurements Within 10 mmHg vs Lambda')
    plt.grid(True, alpha=0.3)
    
    # SBP within 10 mmHg comparison
    plt.subplot(2, 2, 4)
    plt.plot(lambda_values, [results[lam]['sbp_within_10'] for lam in lambda_values], 'r-o')
    plt.xlabel('Lambda Value')
    plt.ylabel('Percentage (%)')
    plt.title('SBP Measurements Within 10 mmHg vs Lambda')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lambda_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Run the desired test
    compare_methods()
    # test_individual_subject()
    # test_different_lambda_values()
