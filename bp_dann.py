import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from flip_gradient import GradientReversal
import pandas as pd


def correlation_coefficient(y_true, y_pred):
    """Calculate Pearson correlation coefficient between true and predicted values"""
    x = y_true
    y = y_pred
    mx = np.mean(x)
    my = np.mean(y)
    xm, ym = x-mx, y-my
    r_num = np.sum(np.multiply(xm, ym))
    r_den = np.sqrt(np.multiply(np.sum(np.square(xm)), np.sum(np.square(ym))))
    
    if r_den == 0:
        return 0
    
    r = r_num / r_den
    r = np.maximum(np.minimum(r, 1.0), -1.0)
    return r


def pad_data(data, scaler, series_len, num_features):
    """Pad sequences to fixed length and apply standardization"""
    data_padded = np.zeros((0, series_len, num_features))
    for i in range(len(data)):
        if len(data[i]) > 0:  # Check if data exists
            l = data[i].shape[0]
            if l > 0:  # Only process if data length > 0
                # Pad with zeros at the beginning and apply scaler
                if l < series_len:
                    padded = np.pad(scaler.transform(data[i]), 
                                    ((series_len-l, 0), (0, 0)), 
                                    'constant', constant_values=-3)
                else:
                    # If sequence is longer than series_len, take the last series_len points
                    padded = scaler.transform(data[i][-series_len:])
                    
                data_padded = np.concatenate((data_padded, 
                                            padded.reshape(1, series_len, num_features)), 
                                            axis=0)
    return data_padded


class RNNModel:
    """Neural network model for blood pressure estimation with domain adaptation"""
    def __init__(self, num_subjects=3, is_training=True):
        # Model hyperparameters
        self.unit_lstm = 30
        self.unit = 30
        self.drop_rate = 0.7 if is_training else 0.0
        
        # Input dimensions
        self.length = 100  # Sequence length
        self.fea_dim = 9   # Input features (4 channels + 4 derivatives + timing)
        self.num_sub = num_subjects  # Number of subjects in domain adaptation
        
        # Create the model architecture
        self.build_model()
        
    def build_model(self):
        """Build the DANN model architecture using Keras functional API"""
        # Input layer
        inputs = Input(shape=(self.length, self.fea_dim), name='input')
        
        # Feature extractor network
        lstm_layer = LSTM(self.unit_lstm, return_sequences=True, name='lstm')(inputs)
        dropout_layer = Dropout(self.drop_rate)(lstm_layer)
        
        # Get last output
        last_output = Lambda(lambda x: x[:, -1, :])(dropout_layer)
        shared_features = Dense(self.unit, activation='relu', name='shared_dense')(last_output)
        
        # Blood pressure estimator network (multitask learning)
        # Diastolic BP path
        dbp_dense1 = Dense(self.unit, activation='relu', name='dbp_dense1')(shared_features)
        dbp_dense2 = Dense(self.unit, activation='relu', name='dbp_dense2')(dbp_dense1)
        dbp_dense3 = Dense(self.unit, activation='relu', name='dbp_dense3')(dbp_dense2)
        dbp_output = Dense(1, name='dbp_out')(dbp_dense3)
        
        # Systolic BP path
        sbp_dense1 = Dense(self.unit, activation='relu', name='sbp_dense1')(shared_features)
        sbp_dense2 = Dense(self.unit, activation='relu', name='sbp_dense2')(sbp_dense1)
        sbp_dense3 = Dense(self.unit, activation='relu', name='sbp_dense3')(sbp_dense2)
        sbp_output = Dense(1, name='sbp_out')(sbp_dense3)
        
        # Domain classifier network (for adversarial training)
        # This part requires the gradient reversal layer
        grl = GradientReversal(1.0)
        lambda_layer = grl(shared_features)
        domain_dense = Dense(30, activation='relu', name='domain1')(lambda_layer)
        domain_output = Dense(self.num_sub, activation='softmax', name='domain_out')(domain_dense)
        
        # Create the complete model
        self.model = Model(inputs=inputs, 
                          outputs=[dbp_output, sbp_output, domain_output])
        
        # Compile the model with appropriate losses and metrics
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),  # Start with a reasonable learning rate
            loss={
                'dbp_out': 'mse',
                'sbp_out': 'mse',
                'domain_out': 'categorical_crossentropy'
            },
            loss_weights={
                'dbp_out': 1.0,
                'sbp_out': 1.0,
                'domain_out': 1.0  # This weight will be adjusted during training
            },
            metrics={
                'dbp_out': 'mse',
                'sbp_out': 'mse',
                'domain_out': 'accuracy'
            }
        )
        
        # Create a separate prediction model without domain output for inference
        self.prediction_model = Model(inputs=inputs, outputs=[dbp_output, sbp_output])


class DataGenerator:
    """Generate batches of data for DANN training"""
    def __init__(self, data_dir='data', minute=4):
        """
        Initialize data generator for DANN training
        
        Args:
            data_dir: Directory containing subject CSV files
            minute: Minutes of data to use for training
        """
        self.data_dir = data_dir
        self.minute = minute
        
        # Load all subject data
        self.data = {}
        self.labels = {}
        
        # Get list of all subject files
        subject_files = [f for f in os.listdir(data_dir) if f.startswith('subject_') and f.endswith('.csv')]
        self.num_subjects = len(subject_files)
        
        print(f"Found {self.num_subjects} subject files")
        
        # Randomly select source, target, and test indices
        all_indices = list(range(1, self.num_subjects + 1))
        random.shuffle(all_indices)
        
        self.source_idx = all_indices[0]
        self.target_idx = all_indices[1]
        self.test_idx = all_indices[2]
        
        subject_list = [self.source_idx, self.target_idx, self.test_idx]
        print(f"Selected subjects - Source: {self.source_idx}, Target: {self.target_idx}, Test: {self.test_idx}")
        
        # Process each subject's data
        for s in subject_list:
            # Load data file
            file_path = os.path.join(data_dir, f"subject_{s:02d}.csv")
            df = pd.read_csv(file_path)
            
            # Extract features and labels
            features = df[['Channel1', 'Channel2', 'Channel3', 'Channel4', 
                          'Channel1_Derivative', 'Channel2_Derivative',
                          'Channel3_Derivative', 'Channel4_Derivative', 'Timing']].values
            
            labels = df[['DBP', 'SBP']].values
            
            # Group by beats
            unique_beats = df['Beat_Number'].unique()
            
            # Create lists to store beat data
            list_data = []
            beat_labels = []
            
            for beat in unique_beats:
                beat_data = features[df['Beat_Number'] == beat]
                if len(beat_data) > 0:
                    list_data.append(beat_data)
                    # Use the first label for this beat (all rows have same label for a beat)
                    beat_labels.append(labels[df['Beat_Number'] == beat][0])
            
            self.data[s] = list_data
            self.labels[s] = np.array(beat_labels)
        
        # Calculate beats for training
        # Count average beats per minute based on test subject
        count = []
        for b in self.data[self.test_idx]:
            count.append(len(b))
        
        avg_len = np.sum(count) / len(count) / 100 if count else 1
        t_beats = int(minute * 60 / avg_len) + 10
        
        print(f"Source beats: {len(self.data[self.source_idx])}, Target beats: {len(self.data[self.target_idx])}")
        print(f"Test beats: {t_beats}, total: {len(count)}, avg len: {avg_len}")
        
        # Use a common batch size for all domains to avoid dimension mismatch
        self.batch_size = 50
        self.batch_num = min(t_beats // self.batch_size, 50)  # Limit to 50 batches to keep training manageable
        
        print(f"Using common batch size: {self.batch_size}")
        print(f"Batch num: {self.batch_num}")
        
        # Create random order indices for each subject
        self.source_order = np.random.permutation(len(self.data[self.source_idx]))
        self.target_order = np.random.permutation(len(self.data[self.target_idx]))
        self.test_order = np.random.permutation(len(self.data[self.test_idx]))
        
        # Normalize labels
        label_all = []
        for ss in self.labels.keys():
            label_all.extend(self.labels[ss])
        
        self.label_scaler = MinMaxScaler(feature_range=(0, 1))
        label_all = np.array(label_all)
        self.label_scaler.fit(label_all)
        
        # Standardize features
        train_data = self.data[self.source_idx]
        scale_data = []
        for beat in train_data:
            if len(beat) > 0:  # Check if beat data exists
                scale_data.extend(beat)
        
        self.standard_data_scaler = StandardScaler()
        if scale_data:  # Check if there is data to fit scaler
            self.standard_data_scaler.fit(scale_data)
        
        # Apply normalization and standardization
        series_len = 100
        num_features = 9
        
        for ss in self.labels.keys():
            self.data[ss] = pad_data(self.data[ss], self.standard_data_scaler, series_len, num_features)
            self.labels[ss] = self.label_scaler.transform(self.labels[ss])
    
    def generate(self, ite, da_train=True):
        """Generate a batch of data for DANN training"""
        batch_size = self.batch_size
        
        # Make sure we don't exceed the array bounds
        source_idx_end = min((ite + 1) * batch_size, len(self.source_order))
        target_idx_end = min((ite + 1) * batch_size, len(self.target_order))
        test_idx_end = min((ite + 1) * batch_size, len(self.test_order))
        
        source_batch_size = source_idx_end - (ite * batch_size)
        target_batch_size = target_idx_end - (ite * batch_size)
        test_batch_size = test_idx_end - (ite * batch_size)
        
        # Initialize empty arrays
        x0 = np.empty((0, self.data[self.source_idx].shape[1], self.data[self.source_idx].shape[2]))
        y0 = np.empty((0, self.labels[self.source_idx].shape[1]))
        x1 = np.empty((0, self.data[self.target_idx].shape[1], self.data[self.target_idx].shape[2]))
        y1 = np.empty((0, self.labels[self.target_idx].shape[1]))
        x2 = np.empty((0, self.data[self.test_idx].shape[1], self.data[self.test_idx].shape[2]))
        y2 = np.empty((0, self.labels[self.test_idx].shape[1]))
        
        if da_train:
            # Source domain data
            if source_batch_size > 0:
                indices = self.source_order[ite * batch_size:source_idx_end]
                x0 = self.data[self.source_idx][indices]
                y0 = self.labels[self.source_idx][indices]
            
            # Target domain data
            if target_batch_size > 0:
                indices = self.target_order[ite * batch_size:target_idx_end]
                x1 = self.data[self.target_idx][indices]
                y1 = self.labels[self.target_idx][indices]
        
        # Test domain data
        if test_batch_size > 0:
            indices = self.test_order[ite * batch_size:test_idx_end]
            x2 = self.data[self.test_idx][indices]
            y2 = self.labels[self.test_idx][indices]
        
        return x0, y0, x1, y1, x2, y2


def train_and_evaluate(data_dir='data', minutes=4, num_steps=400, epochs=50):
    """
    Train DANN model and evaluate performance
    
    Args:
        data_dir: Directory containing subject CSV files
        minutes: Minutes of data to use for training
        num_steps: Total number of steps for domain adaptation
        epochs: Number of training epochs
    """
    # Set random seeds for reproducibility
    random.seed(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    
    # Initialize data generator
    data_generator = DataGenerator(data_dir, minutes)
    
    # Create the model
    rnn_model = RNNModel(num_subjects=3)
    
    # Create directories for saving models and results
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Domain adaptation training
    da_batch_num = data_generator.batch_num
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Adaptation parameter schedule - gradually increases importance of domain classification
        p = float(epoch) / num_steps
        l = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1
        lr = 0.01 / (1. + 10 * p)**0.75      # gradually decrease learning rate
        
        # Create a new model with updated learning rate and domain weight for this epoch
        if epoch > 0:  # Only update after first epoch
            # Save weights
            weights = rnn_model.model.get_weights()
            
            # Rebuild model with new learning rate
            rnn_model = RNNModel(num_subjects=3)
            rnn_model.model.set_weights(weights)
            
            # Recompile with new learning rate
            rnn_model.model.compile(
                optimizer=Adam(learning_rate=float(lr)),
                loss={
                    'dbp_out': 'mse',
                    'sbp_out': 'mse',
                    'domain_out': 'categorical_crossentropy'
                },
                loss_weights={
                    'dbp_out': 1.0,
                    'sbp_out': 1.0,
                    'domain_out': float(l)  # Update domain loss weight
                },
                metrics={
                    'dbp_out': 'mse',
                    'sbp_out': 'mse',
                    'domain_out': 'accuracy'
                }
            )
        
        epoch_losses = []
        
        # Process batches
        for b in range(da_batch_num):
            x0, y0, x1, y1, x2, y2 = data_generator.generate(b)
            
            # Process source domain
            if len(x0) > 0:
                domain_labels = np.zeros((len(x0), 3))
                domain_labels[:, 0] = 1  # Source domain
                
                y_dbp = np.expand_dims(y0[:, 0], -1)
                y_sbp = np.expand_dims(y0[:, 1], -1)
                
                batch_loss_source = rnn_model.model.train_on_batch(
                    x0, 
                    [y_dbp, y_sbp, domain_labels],
                    return_dict=True
                )
                epoch_losses.append(batch_loss_source['loss'])
            
            # Process target domain
            if len(x1) > 0:
                domain_labels = np.zeros((len(x1), 3))
                domain_labels[:, 1] = 1  # Target domain
                
                y_dbp = np.expand_dims(y1[:, 0], -1)
                y_sbp = np.expand_dims(y1[:, 1], -1)
                
                batch_loss_target = rnn_model.model.train_on_batch(
                    x1, 
                    [y_dbp, y_sbp, domain_labels],
                    return_dict=True
                )
                epoch_losses.append(batch_loss_target['loss'])
            
            # Process test domain
            if len(x2) > 0:
                domain_labels = np.zeros((len(x2), 3))
                domain_labels[:, 2] = 1  # Test domain
                
                y_dbp = np.expand_dims(y2[:, 0], -1)
                y_sbp = np.expand_dims(y2[:, 1], -1)
                
                batch_loss_test = rnn_model.model.train_on_batch(
                    x2, 
                    [y_dbp, y_sbp, domain_labels],
                    return_dict=True
                )
                epoch_losses.append(batch_loss_test['loss'])
            
            if b % 10 == 0:
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                print(f"  Batch {b}/{da_batch_num}, Avg Loss: {avg_loss:.4f}")
        
        # Print training progress
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        print(f"  Epoch {epoch+1} Avg Loss: {avg_loss:.4f}, LR: {lr:.6f}, Lambda: {l:.4f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = f"{checkpoint_dir}/dann_{data_generator.test_idx}_{minutes}min_epoch{epoch+1}.keras"
            rnn_model.model.save(save_path)
            print(f"  Model checkpoint saved to {save_path}")
    
    # Save final model
    save_path = f"{checkpoint_dir}/dann_{data_generator.test_idx}_{minutes}min_final.keras"
    rnn_model.model.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Evaluate on test subject
    all_test_x = []
    all_test_y = []
    
    # Collect all test data
    for b in range(da_batch_num):
        _, _, _, _, x2, y2 = data_generator.generate(b, da_train=False)
        if len(x2) > 0:
            all_test_x.append(x2)
            all_test_y.append(y2)
    
    if all_test_x:
        test_x = np.vstack(all_test_x)
        test_y = np.vstack(all_test_y)
        
        # Get model predictions
        dbp_pred, sbp_pred = rnn_model.prediction_model.predict(test_x)
        
        # Transform predictions back to original scale
        y_pred = np.hstack([dbp_pred, sbp_pred])
        y_pred = data_generator.label_scaler.inverse_transform(y_pred)
        test_y = data_generator.label_scaler.inverse_transform(test_y)
        
        # Calculate metrics
        dbp_rmse = np.sqrt(mean_squared_error(test_y[:, 0], y_pred[:, 0]))
        sbp_rmse = np.sqrt(mean_squared_error(test_y[:, 1], y_pred[:, 1]))
        
        dbp_r = correlation_coefficient(test_y[:, 0], y_pred[:, 0])
        sbp_r = correlation_coefficient(test_y[:, 1], y_pred[:, 1])
        
        print("\nTest Results:")
        print(f"DBP - RMSE: {dbp_rmse:.2f} mmHg, Correlation: {dbp_r:.2f}")
        print(f"SBP - RMSE: {sbp_rmse:.2f} mmHg, Correlation: {sbp_r:.2f}")
        
        # Calculate percentage of estimates within 10 mmHg of reference
        dbp_within_10 = np.mean(np.abs(test_y[:, 0] - y_pred[:, 0]) < 10) * 100
        sbp_within_10 = np.mean(np.abs(test_y[:, 1] - y_pred[:, 1]) < 10) * 100
        
        print(f"DBP - Within 10 mmHg: {dbp_within_10:.1f}%")
        print(f"SBP - Within 10 mmHg: {sbp_within_10:.1f}%")
        
        # Plot Bland-Altman plots
        plt.figure(figsize=(12, 5))
        
        # DBP Bland-Altman
        plt.subplot(1, 2, 1)
        mean_dbp = (test_y[:, 0] + y_pred[:, 0]) / 2
        diff_dbp = test_y[:, 0] - y_pred[:, 0]
        md_dbp = np.mean(diff_dbp)
        sd_dbp = np.std(diff_dbp)
        
        plt.scatter(mean_dbp, diff_dbp, alpha=0.5)
        plt.axhline(md_dbp, color='k', linestyle='-')
        plt.axhline(md_dbp + 1.96*sd_dbp, color='k', linestyle='--')
        plt.axhline(md_dbp - 1.96*sd_dbp, color='k', linestyle='--')
        plt.title(f'DBP Bland-Altman (RMSE: {dbp_rmse:.2f}, r: {dbp_r:.2f})')
        plt.xlabel('Mean DBP (mmHg)')
        plt.ylabel('Difference (mmHg)')
        
        # SBP Bland-Altman
        plt.subplot(1, 2, 2)
        mean_sbp = (test_y[:, 1] + y_pred[:, 1]) / 2
        diff_sbp = test_y[:, 1] - y_pred[:, 1]
        md_sbp = np.mean(diff_sbp)
        sd_sbp = np.std(diff_sbp)
        
        plt.scatter(mean_sbp, diff_sbp, alpha=0.5)
        plt.axhline(md_sbp, color='k', linestyle='-')
        plt.axhline(md_sbp + 1.96*sd_sbp, color='k', linestyle='--')
        plt.axhline(md_sbp - 1.96*sd_sbp, color='k', linestyle='--')
        plt.title(f'SBP Bland-Altman (RMSE: {sbp_rmse:.2f}, r: {sbp_r:.2f})')
        plt.xlabel('Mean SBP (mmHg)')
        plt.ylabel('Difference (mmHg)')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/bland_altman_{data_generator.test_idx}_{minutes}min.png', dpi=300)
        plt.close()
        
        # Also create prediction vs. reference scatter plots
        plt.figure(figsize=(12, 5))
        
        # DBP scatter
        plt.subplot(1, 2, 1)
        plt.scatter(test_y[:, 0], y_pred[:, 0], alpha=0.5)
        min_val = min(test_y[:, 0].min(), y_pred[:, 0].min())
        max_val = max(test_y[:, 0].max(), y_pred[:, 0].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.title(f'DBP Prediction vs Reference (r: {dbp_r:.2f})')
        plt.xlabel('Reference DBP (mmHg)')
        plt.ylabel('Predicted DBP (mmHg)')
        
        # SBP scatter
        plt.subplot(1, 2, 2)
        plt.scatter(test_y[:, 1], y_pred[:, 1], alpha=0.5)
        min_val = min(test_y[:, 1].min(), y_pred[:, 1].min())
        max_val = max(test_y[:, 1].max(), y_pred[:, 1].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.title(f'SBP Prediction vs Reference (r: {sbp_r:.2f})')
        plt.xlabel('Reference SBP (mmHg)')
        plt.ylabel('Predicted SBP (mmHg)')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/scatter_{data_generator.test_idx}_{minutes}min.png', dpi=300)
        plt.close()
        
        # Save results to CSV
        results = {
            'subject': data_generator.test_idx,
            'minutes': minutes,
            'dbp_rmse': dbp_rmse,
            'sbp_rmse': sbp_rmse,
            'dbp_r': dbp_r,
            'sbp_r': sbp_r,
            'dbp_within_10': dbp_within_10,
            'sbp_within_10': sbp_within_10
        }
        
        # Save results to a CSV file
        results_file = f'{results_dir}/results.csv'
        if not os.path.exists(results_file):
            with open(results_file, 'w') as f:
                f.write('subject,minutes,dbp_rmse,sbp_rmse,dbp_r,sbp_r,dbp_within_10,sbp_within_10\n')
        
        with open(results_file, 'a') as f:
            f.write(f"{results['subject']},{results['minutes']},{results['dbp_rmse']:.2f},{results['sbp_rmse']:.2f},"
                   f"{results['dbp_r']:.2f},{results['sbp_r']:.2f},{results['dbp_within_10']:.1f},{results['sbp_within_10']:.1f}\n")
        
        return results