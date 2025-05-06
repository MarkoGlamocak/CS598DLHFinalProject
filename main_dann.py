import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import flip_gradient
import random
import os

# Assuming the data_preprocess function is imported from data_preparation.py
from data_preparation import data_preprocess

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

class DomainAdversarialModel:
    def __init__(self, lstm_units=30, dropout_rate=0.7, learning_rate=0.003):
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.feature_extractor = None
        self.bp_estimator_dbp = None
        self.bp_estimator_sbp = None
        self.domain_classifier = None
        self.combined_model = None
        
    def build_model(self, input_shape, num_domains):
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Feature Extractor
        lstm = layers.LSTM(self.lstm_units, return_sequences=False)(inputs)
        lstm_dropout = layers.Dropout(self.dropout_rate)(lstm)
        shared_features = layers.Dense(20, activation='relu')(lstm_dropout)
        
        # Blood Pressure Estimators
        # DBP Estimator
        dbp_dense1 = layers.Dense(15, activation='relu')(shared_features)
        dbp_dense2 = layers.Dense(10, activation='relu')(dbp_dense1)
        dbp_output = layers.Dense(1, name='dbp_output')(dbp_dense2)
        
        # SBP Estimator
        sbp_dense1 = layers.Dense(15, activation='relu')(shared_features)
        sbp_dense2 = layers.Dense(10, activation='relu')(sbp_dense1)
        sbp_output = layers.Dense(1, name='sbp_output')(sbp_dense2)
        
        # Domain Classifier with Gradient Reversal Layer
        grl = flip_gradient.GradientReversal(lambda_=1.0)(shared_features)
        domain_dense1 = layers.Dense(15, activation='relu')(grl)
        domain_dense2 = layers.Dense(10, activation='relu')(domain_dense1)
        domain_output = layers.Dense(num_domains, activation='softmax', name='domain_output')(domain_dense2)
        
        # Combined Model
        self.combined_model = models.Model(inputs=inputs, outputs=[dbp_output, sbp_output, domain_output])
        
        # Compile the model with appropriate loss functions and metrics
        self.combined_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'dbp_output': 'mse',
                'sbp_output': 'mse',
                'domain_output': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'dbp_output': 1.0,
                'sbp_output': 1.0,
                'domain_output': 0.1  # Weight for domain classification loss
            },
            metrics={
                'dbp_output': 'mse',
                'sbp_output': 'mse',
                'domain_output': 'accuracy'
            }
        )
        
        return self.combined_model
    
    def train(self, data_generator, epochs=100, validation_data=None):
        # Callbacks for adaptive learning rate and early stopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train the model
        history = self.combined_model.fit(
            data_generator,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[reduce_lr, early_stopping]
        )
        
        return history
    
    def evaluate(self, X_test, y_dbp_test, y_sbp_test):
        # Predict BP values
        dbp_pred, sbp_pred, _ = self.combined_model.predict(X_test)
        
        # Calculate RMSE
        dbp_rmse = np.sqrt(np.mean((dbp_pred.flatten() - y_dbp_test) ** 2))
        sbp_rmse = np.sqrt(np.mean((sbp_pred.flatten() - y_sbp_test) ** 2))
        
        # Calculate correlation coefficient
        dbp_corr, _ = pearsonr(dbp_pred.flatten(), y_dbp_test)
        sbp_corr, _ = pearsonr(sbp_pred.flatten(), y_sbp_test)
        
        print(f"DBP RMSE: {dbp_rmse:.2f} mmHg, Correlation: {dbp_corr:.2f}")
        print(f"SBP RMSE: {sbp_rmse:.2f} mmHg, Correlation: {sbp_corr:.2f}")
        
        # Calculate percentage of measurements within 10 mmHg
        dbp_within_10 = np.mean(np.abs(dbp_pred.flatten() - y_dbp_test) <= 10) * 100
        sbp_within_10 = np.mean(np.abs(sbp_pred.flatten() - y_sbp_test) <= 10) * 100
        
        print(f"DBP measurements within 10 mmHg: {dbp_within_10:.2f}%")
        print(f"SBP measurements within 10 mmHg: {sbp_within_10:.2f}%")
        
        # Return evaluation metrics
        return {
            'dbp_rmse': dbp_rmse,
            'sbp_rmse': sbp_rmse,
            'dbp_corr': dbp_corr,
            'sbp_corr': sbp_corr,
            'dbp_within_10': dbp_within_10,
            'sbp_within_10': sbp_within_10,
            'dbp_pred': dbp_pred.flatten(),
            'sbp_pred': sbp_pred.flatten(),
            'dbp_true': y_dbp_test,
            'sbp_true': y_sbp_test
        }


def plot_consolidated_bland_altman(all_runs_metrics, test_subject_idx, test_mins, save_dir='results'):
    """
    Generate consolidated Bland-Altman plots for all runs of a specific duration.
    
    Parameters:
    all_runs_metrics: List of metrics dictionaries from all runs
    test_subject_idx: Subject ID for the test subject
    test_mins: Number of minutes used for training (3, 4, or 5)
    save_dir: Directory to save the plots
    """
    plt.ioff()
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_dbp_pred = []
    all_dbp_true = []
    all_sbp_pred = []
    all_sbp_true = []
    
    for metrics in all_runs_metrics:
        all_dbp_pred.extend(metrics['dbp_pred'])
        all_dbp_true.extend(metrics['dbp_true'])
        all_sbp_pred.extend(metrics['sbp_pred'])
        all_sbp_true.extend(metrics['sbp_true'])
    
    all_dbp_pred = np.array(all_dbp_pred)
    all_dbp_true = np.array(all_dbp_true)
    all_sbp_pred = np.array(all_sbp_pred)
    all_sbp_true = np.array(all_sbp_true)
    
    dbp_mean = (all_dbp_pred + all_dbp_true) / 2
    dbp_diff = all_dbp_pred - all_dbp_true
    dbp_mean_diff = np.mean(dbp_diff)
    dbp_std_diff = np.std(dbp_diff)
    
    sbp_mean = (all_sbp_pred + all_sbp_true) / 2
    sbp_diff = all_sbp_pred - all_sbp_true
    sbp_mean_diff = np.mean(sbp_diff)
    sbp_std_diff = np.std(sbp_diff)
    
    plt.figure(figsize=(12, 5))

    # DBP Bland-Altman
    plt.subplot(1, 2, 1)
    plt.scatter(dbp_mean, dbp_diff, alpha=0.3, s=10)
    plt.axhline(y=dbp_mean_diff, color='r', linestyle='-', label=f'Mean: {dbp_mean_diff:.2f}')
    plt.axhline(y=dbp_mean_diff + 1.96 * dbp_std_diff, color='g', linestyle='--', 
                label=f'+1.96 SD: {dbp_mean_diff + 1.96 * dbp_std_diff:.2f}')
    plt.axhline(y=dbp_mean_diff - 1.96 * dbp_std_diff, color='g', linestyle='--', 
                label=f'-1.96 SD: {dbp_mean_diff - 1.96 * dbp_std_diff:.2f}')
    plt.xlabel('Mean of Predicted and True DBP (mmHg)')
    plt.ylabel('Difference (Predicted - True) DBP (mmHg)')
    plt.title(f'Bland-Altman Plot for DBP - {test_mins} minutes training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SBP Bland-Altman
    plt.subplot(1, 2, 2)
    plt.scatter(sbp_mean, sbp_diff, alpha=0.3, s=10)  # Smaller points with more transparency
    plt.axhline(y=sbp_mean_diff, color='r', linestyle='-', label=f'Mean: {sbp_mean_diff:.2f}')
    plt.axhline(y=sbp_mean_diff + 1.96 * sbp_std_diff, color='g', linestyle='--', 
                label=f'+1.96 SD: {sbp_mean_diff + 1.96 * sbp_std_diff:.2f}')
    plt.axhline(y=sbp_mean_diff - 1.96 * sbp_std_diff, color='g', linestyle='--', 
                label=f'-1.96 SD: {sbp_mean_diff - 1.96 * sbp_std_diff:.2f}')
    plt.xlabel('Mean of Predicted and True SBP (mmHg)')
    plt.ylabel('Difference (Predicted - True) SBP (mmHg)')
    plt.title(f'Bland-Altman Plot for SBP - {test_mins} minutes training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'consolidated_bland_altman_subject{test_subject_idx}_{test_mins}mins.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Consolidated Bland-Altman plot saved to {save_path}")
    
    return save_path


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_source, y_dbp_source, y_sbp_source, X_target, y_dbp_target, y_sbp_target, 
                 X_test=None, y_dbp_test=None, y_sbp_test=None, batch_size=100, test_mins=4, shuffle=True):
        """
        DataGenerator for DANN model that creates balanced batches from source and target domains.
        
        Parameters:
        X_source, y_dbp_source, y_sbp_source: Data from source domain (subject 1)
        X_target, y_dbp_target, y_sbp_target: Data from target domain (subject 2)
        X_test, y_dbp_test, y_sbp_test: Data from test domain (subject 3) with limited training data
        batch_size: Batch size for source and target domains
        test_mins: Minutes of data to use from test subject (3, 4, or 5)
        shuffle: Whether to shuffle the data each epoch
        """
        self.X_source = X_source
        self.y_dbp_source = y_dbp_source
        self.y_sbp_source = y_sbp_source
        
        self.X_target = X_target
        self.y_dbp_target = y_dbp_target
        self.y_sbp_target = y_sbp_target
        
        # If test data is provided, use it
        self.use_test_data = X_test is not None
        if self.use_test_data:
            # Convert minutes to number of samples (assuming 1 sample per second)
            test_samples = test_mins * 60
            
            # Limit test data to the specified number of minutes
            self.X_test = X_test[:test_samples]
            self.y_dbp_test = y_dbp_test[:test_samples]
            self.y_sbp_test = y_sbp_test[:test_samples]
            
            # Adjust batch size for test domain to ensure balanced batches
            self.test_batch_size = max(1, min(len(self.X_test), batch_size // 2))
        else:
            self.X_test = None
            self.y_dbp_test = None
            self.y_sbp_test = None
            self.test_batch_size = 0
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Domain labels (0 for source, 1 for target, 2 for test)
        self.domain_labels_source = np.zeros(len(self.X_source))
        self.domain_labels_target = np.ones(len(self.X_target))
        if self.use_test_data:
            self.domain_labels_test = np.full(len(self.X_test), 2)
        
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        # Calculate total samples needed and divide by effective batch size
        source_batches = len(self.X_source) // (self.batch_size // 2)
        target_batches = len(self.X_target) // (self.batch_size // 2)
        
        if self.use_test_data:
            test_batches = len(self.X_test) // self.test_batch_size
            return min(source_batches, target_batches, test_batches)
        else:
            return min(source_batches, target_batches)
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Calculate start and end indices for this batch
        source_start = (index * self.batch_size // 2) % len(self.indices_source)
        source_end = min(source_start + self.batch_size // 2, len(self.indices_source))
        source_indices = self.indices_source[source_start:source_end]
        
        target_start = (index * self.batch_size // 2) % len(self.indices_target)
        target_end = min(target_start + self.batch_size // 2, len(self.indices_target))
        target_indices = self.indices_target[target_start:target_end]
        
        # Get data for source and target domains
        X_source_batch = self.X_source[source_indices]
        y_dbp_source_batch = self.y_dbp_source[source_indices]
        y_sbp_source_batch = self.y_sbp_source[source_indices]
        domain_labels_source_batch = self.domain_labels_source[source_indices]
        
        X_target_batch = self.X_target[target_indices]
        y_dbp_target_batch = self.y_dbp_target[target_indices]
        y_sbp_target_batch = self.y_sbp_target[target_indices]
        domain_labels_target_batch = self.domain_labels_target[target_indices]
        
        # Combine source and target data
        X_batch = np.concatenate([X_source_batch, X_target_batch], axis=0)
        y_dbp_batch = np.concatenate([y_dbp_source_batch, y_dbp_target_batch], axis=0)
        y_sbp_batch = np.concatenate([y_sbp_source_batch, y_sbp_target_batch], axis=0)
        domain_labels_batch = np.concatenate([domain_labels_source_batch, domain_labels_target_batch], axis=0)
        
        # If test data is available, add it to the batch
        if self.use_test_data:
            test_start = (index * self.test_batch_size) % len(self.indices_test)
            test_end = min(test_start + self.test_batch_size, len(self.indices_test))
            test_indices = self.indices_test[test_start:test_end]
            
            X_test_batch = self.X_test[test_indices]
            y_dbp_test_batch = self.y_dbp_test[test_indices]
            y_sbp_test_batch = self.y_sbp_test[test_indices]
            domain_labels_test_batch = self.domain_labels_test[test_indices]
            
            X_batch = np.concatenate([X_batch, X_test_batch], axis=0)
            y_dbp_batch = np.concatenate([y_dbp_batch, y_dbp_test_batch], axis=0)
            y_sbp_batch = np.concatenate([y_sbp_batch, y_sbp_test_batch], axis=0)
            domain_labels_batch = np.concatenate([domain_labels_batch, domain_labels_test_batch], axis=0)
        
        # Shuffle the batch to mix domains
        indices = np.arange(len(X_batch))
        np.random.shuffle(indices)
        X_batch = X_batch[indices]
        y_dbp_batch = y_dbp_batch[indices]
        y_sbp_batch = y_sbp_batch[indices]
        domain_labels_batch = domain_labels_batch[indices]
        
        return X_batch, {'dbp_output': y_dbp_batch, 'sbp_output': y_sbp_batch, 'domain_output': domain_labels_batch}
    
    def on_epoch_end(self):
        """Updates indices after each epoch"""
        self.indices_source = np.arange(len(self.X_source))
        self.indices_target = np.arange(len(self.X_target))
        if self.use_test_data:
            self.indices_test = np.arange(len(self.X_test))
        
        if self.shuffle:
            np.random.shuffle(self.indices_source)
            np.random.shuffle(self.indices_target)
            if self.use_test_data:
                np.random.shuffle(self.indices_test)


def run_experiment(subjects_data, test_subject_idx, source_subject_idx=None, target_subject_idx=None, 
                   test_mins=4, epochs=100, batch_size=100):
    """
    Run a DANN experiment with specified subjects and training durations.
    
    Parameters:
    subjects_data: Dictionary of tuples (X, y_dbp, y_sbp) for each subject
    test_subject_idx: Key of the subject to test on (with limited training data)
    source_subject_idx: Key of the source domain subject (if None, use first non-test subject)
    target_subject_idx: Key of the target domain subject (if None, use second non-test subject)
    test_mins: Minutes of data to use from test subject (3, 4, or 5)
    epochs: Number of training epochs
    batch_size: Batch size for training
    
    Returns:
    Evaluation metrics for the test subject
    """
    # If source and target subjects not specified, select them automatically
    if source_subject_idx is None or target_subject_idx is None:
        available_indices = [i for i in subjects_data.keys() if i != test_subject_idx]
        if len(available_indices) < 2:
            raise ValueError("Need at least 3 subjects for this experiment")
        
        if source_subject_idx is None:
            source_subject_idx = available_indices[0]
        
        if target_subject_idx is None:
            target_subject_idx = available_indices[1]
    
    # Get data for each subject
    X_test, y_dbp_test, y_sbp_test = subjects_data[test_subject_idx]
    X_source, y_dbp_source, y_sbp_source = subjects_data[source_subject_idx]
    X_target, y_dbp_target, y_sbp_target = subjects_data[target_subject_idx]
    
    # Ensure data is in proper numpy array format
    X_test = np.asarray(X_test) if not isinstance(X_test, np.ndarray) else X_test
    y_dbp_test = np.asarray(y_dbp_test) if not isinstance(y_dbp_test, np.ndarray) else y_dbp_test
    y_sbp_test = np.asarray(y_sbp_test) if not isinstance(y_sbp_test, np.ndarray) else y_sbp_test
    
    X_source = np.asarray(X_source) if not isinstance(X_source, np.ndarray) else X_source
    y_dbp_source = np.asarray(y_dbp_source) if not isinstance(y_dbp_source, np.ndarray) else y_dbp_source
    y_sbp_source = np.asarray(y_sbp_source) if not isinstance(y_sbp_source, np.ndarray) else y_sbp_source
    
    X_target = np.asarray(X_target) if not isinstance(X_target, np.ndarray) else X_target
    y_dbp_target = np.asarray(y_dbp_target) if not isinstance(y_dbp_target, np.ndarray) else y_dbp_target
    y_sbp_target = np.asarray(y_sbp_target) if not isinstance(y_sbp_target, np.ndarray) else y_sbp_target
    
    # Check if arrays are properly shaped, if not create synthetic data
    if X_test.ndim == 0 or X_source.ndim == 0 or X_target.ndim == 0:
        print("Warning: Creating synthetic data for testing as actual data is not properly shaped")
        sequence_length = 10  # Adjust based on your expected sequence length
        features = 9  # 4 channels, 4 derivatives, and timing
        sample_count = 1000  # Number of samples to generate
        
        # Create synthetic data with appropriate shapes
        X_test = np.random.rand(sample_count, sequence_length, features)
        y_dbp_test = np.random.rand(sample_count)
        y_sbp_test = np.random.rand(sample_count)
        
        X_source = np.random.rand(sample_count, sequence_length, features)
        y_dbp_source = np.random.rand(sample_count)
        y_sbp_source = np.random.rand(sample_count)
        
        X_target = np.random.rand(sample_count, sequence_length, features)
        y_dbp_target = np.random.rand(sample_count)
        y_sbp_target = np.random.rand(sample_count)
    
    # Split test subject data into train (limited) and test sets
    test_train_samples = test_mins * 60  # Convert minutes to samples (assuming 1 sample per second)
    
    # Ensure we don't try to index beyond the length of the arrays
    test_train_samples = min(test_train_samples, len(X_test) - 1)
    
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
        batch_size=batch_size,
        test_mins=test_mins
    )
    
    # Build and train the model
    input_shape = X_source.shape[1:]
    num_domains = 3  # source, target, and test domains
    
    dann_model = DomainAdversarialModel()
    model = dann_model.build_model(input_shape, num_domains)
    
    print(f"Training with {test_mins} minutes of data from test subject...")
    history = dann_model.train(data_gen, epochs=epochs)
    
    # Evaluate on test subject's test set
    print(f"Evaluating on test subject (Subject {test_subject_idx})...")
    metrics = dann_model.evaluate(X_test_eval, y_dbp_test_eval, y_sbp_test_eval)
    
    return metrics, history, dann_model


def main():
    # Set seeds for reproducibility
    set_seeds(42)

    results_dir = 'bland_altman_plots'
    os.makedirs(results_dir, exist_ok=True)

    # Load and preprocess data for all subjects
    subjects_data = data_preprocess()
    
    # Test different durations of training data
    test_durations = [3, 4, 5]  # minutes
    
    # Number of runs for robustness
    num_runs = 10
    
    # Store results for all subjects
    all_results = {}
    
    # Get all subject IDs
    all_subject_ids = list(subjects_data.keys())
    print(f"Testing {len(all_subject_ids)} subjects")
    
    # Test each subject as the test subject
    for test_subject_idx in all_subject_ids:
        print(f"\n===== Testing Subject {test_subject_idx} as Test Subject =====")
        
        # Results for this subject across different training durations and runs
        subject_results = {}

        consolidated_metrics_by_duration = {
            3: [],
            4: [],
            5: []
        }
        
        for test_mins in test_durations:
            print(f"\n=== Testing with {test_mins} minutes of training data ===")
            
            # Initialize results for this duration
            duration_results = []
            
            # Get available subjects to use as source and target
            available_indices = [i for i in all_subject_ids if i != test_subject_idx]
            
            # Make sure we have at least 2 other subjects
            if len(available_indices) < 2:
                print("Need at least 3 subjects for this experiment")
                continue
            
            # Run the test multiple times for robustness
            for run in range(num_runs):
                print(f"Run {run+1}/{num_runs}")
                
                # Randomly select source and target domains
                selected_indices = random.sample(available_indices, 2)
                source_subject_idx = selected_indices[0]
                target_subject_idx = selected_indices[1]
                
                print(f"Using Subject {source_subject_idx} as source and Subject {target_subject_idx} as target")
                
                metrics, _, _ = run_experiment(
                    subjects_data, 
                    test_subject_idx=test_subject_idx,
                    source_subject_idx=source_subject_idx,
                    target_subject_idx=target_subject_idx,
                    test_mins=test_mins,
                    epochs=100,
                    batch_size=100
                )
                
                duration_results.append(metrics)
                
                # Store metrics for consolidated Bland-Altman plot
                consolidated_metrics_by_duration[test_mins].append(metrics)
            
            # Calculate average metrics across all runs
            avg_metrics = {
                'dbp_rmse': np.mean([m['dbp_rmse'] for m in duration_results]),
                'sbp_rmse': np.mean([m['sbp_rmse'] for m in duration_results]),
                'dbp_corr': np.mean([m['dbp_corr'] for m in duration_results]),
                'sbp_corr': np.mean([m['sbp_corr'] for m in duration_results]),
                'dbp_within_10': np.mean([m['dbp_within_10'] for m in duration_results]),
                'sbp_within_10': np.mean([m['sbp_within_10'] for m in duration_results]),
                'individual_runs': duration_results
            }
            
            subject_results[test_mins] = avg_metrics

            plot_consolidated_bland_altman(
                consolidated_metrics_by_duration[test_mins],
                test_subject_idx,
                test_mins,
                save_dir=results_dir
            )
        
        # Store results for this subject
        all_results[test_subject_idx] = subject_results
    
    with open('dann_results_summary.txt', 'w') as file:
        for test_mins in test_durations:
            file.write(f"\n===== Summary of Average Results Across 10 Runs for All Subjects (Training Minutes = {test_mins}) =====\n")
            file.write("Subject | Training Minutes | DBP RMSE | SBP RMSE | DBP R | SBP R | DBP % within 10mmHg | SBP % within 10mmHg\n")
            file.write("------- | ---------------- | -------- | -------- | ----- | ----- | ------------------- | -------------------\n")
            
            all_dbp_rmse = []
            all_sbp_rmse = []
            all_dbp_corr = []
            all_sbp_corr = []
            all_dbp_within_10 = []
            all_sbp_within_10 = []
            
            for subject_idx, results in all_results.items():
                if test_mins in results:
                    metrics = results[test_mins]
                    file.write(f"{subject_idx:7d} |  {test_mins:15d} | {metrics['dbp_rmse']:8.2f} | {metrics['sbp_rmse']:8.2f} | "
                          f"{metrics['dbp_corr']:5.2f} | {metrics['sbp_corr']:5.2f} | "
                          f" {metrics['dbp_within_10']:18.2f} |  {metrics['sbp_within_10']:18.2f}\n")
                    
                    all_dbp_rmse.append(metrics['dbp_rmse'])
                    all_sbp_rmse.append(metrics['sbp_rmse'])
                    all_dbp_corr.append(metrics['dbp_corr'])
                    all_sbp_corr.append(metrics['sbp_corr'])
                    all_dbp_within_10.append(metrics['dbp_within_10'])
                    all_sbp_within_10.append(metrics['sbp_within_10'])
            
            mean_dbp_rmse = np.mean(all_dbp_rmse)
            mean_sbp_rmse = np.mean(all_sbp_rmse)
            mean_dbp_corr = np.mean(all_dbp_corr)
            mean_sbp_corr = np.mean(all_sbp_corr)
            mean_dbp_within_10 = np.mean(all_dbp_within_10)
            mean_sbp_within_10 = np.mean(all_sbp_within_10)
            
            file.write("   Mean |  {0:15d} | {1:8.2f} | {2:8.2f} | {3:5.2f} | {4:5.2f} |  {5:18.2f} |  {6:18.2f}\n".format(
                test_mins, mean_dbp_rmse, mean_sbp_rmse, mean_dbp_corr, mean_sbp_corr, 
                mean_dbp_within_10, mean_sbp_within_10))
    
    print("Results have been saved to 'dann_results_summary.txt'")
    print(f"Consolidated Bland-Altman plots have been saved to the '{results_dir}' directory")

if __name__ == "__main__":
    main()
