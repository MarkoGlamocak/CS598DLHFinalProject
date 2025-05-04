import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import random
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class GradientReversalLayer(tf.keras.layers.Layer):
    """
    Custom layer that multiplies the gradient by -1 during backpropagation.
    This is a key component for Domain-Adversarial Neural Networks.
    """
    def __init__(self, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        return inputs
    
    def get_config(self):
        config = super(GradientReversalLayer, self).get_config()
        return config
    
    # Custom gradient
    @tf.custom_gradient
    def gradient_reversal(self, x):
        def grad(dy):
            return -1 * dy
        return x, grad
    
    def call(self, inputs):
        return self.gradient_reversal(inputs)


class BPEstimationModel:
    """
    Base class for blood pressure estimation models.
    
    This implements the MTL model for beat-to-beat blood pressure estimation
    using bioimpedance signals.
    """
    def __init__(self, input_shape=(9,), lstm_units=64, hidden_size=30, dropout_rate=0.2):
        """
        Initialize BP estimation model.
        
        Args:
            input_shape: Shape of input features
            lstm_units: Number of LSTM units
            hidden_size: Size of hidden layers in task-specific networks
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Build and compile the model
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build the base MTL model for blood pressure estimation.
        """
        # Input layer
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Feature extractor
        x = tf.keras.layers.Reshape((self.input_shape[0], 1))(inputs)  # Reshape for LSTM
        x = tf.keras.layers.LSTM(self.lstm_units)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        shared = tf.keras.layers.Dense(self.hidden_size, activation='relu')(x)
        
        # Task-specific networks for DBP and SBP
        # DBP estimation network
        dbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(shared)
        dbp = tf.keras.layers.Dropout(self.dropout_rate)(dbp)
        dbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(dbp)
        dbp = tf.keras.layers.Dropout(self.dropout_rate)(dbp)
        dbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(dbp)
        dbp = tf.keras.layers.Dropout(self.dropout_rate)(dbp)
        dbp_output = tf.keras.layers.Dense(1, name='dbp_output')(dbp)
        
        # SBP estimation network
        sbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(shared)
        sbp = tf.keras.layers.Dropout(self.dropout_rate)(sbp)
        sbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(sbp)
        sbp = tf.keras.layers.Dropout(self.dropout_rate)(sbp)
        sbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(sbp)
        sbp = tf.keras.layers.Dropout(self.dropout_rate)(sbp)
        sbp_output = tf.keras.layers.Dense(1, name='sbp_output')(sbp)
        
        # Create model
        model = tf.keras.models.Model(inputs=inputs, outputs=[dbp_output, sbp_output])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'dbp_output': 'mse',
                'sbp_output': 'mse'
            }
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=1):
        """
        Train the MTL model.
        
        Args:
            X_train: Training features
            y_train: Training labels (DBP, SBP)
            X_val: Validation features
            y_val: Validation labels (DBP, SBP)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level for training
            
        Returns:
            History of training
        """
        # Prepare labels
        y_train_dbp = y_train[:, 0:1]
        y_train_sbp = y_train[:, 1:2]
        y_val_dbp = y_val[:, 0:1]
        y_val_sbp = y_val[:, 1:2]
        
        # Train model
        history = self.model.fit(
            X_train,
            {'dbp_output': y_train_dbp, 'sbp_output': y_train_sbp},
            validation_data=(
                X_val,
                {'dbp_output': y_val_dbp, 'sbp_output': y_val_sbp}
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels (DBP, SBP)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare labels
        y_test_dbp = y_test[:, 0:1]
        y_test_sbp = y_test[:, 1:2]
        
        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_dbp = y_pred[0]
        y_pred_sbp = y_pred[1]
        
        # Calculate metrics
        dbp_rmse = np.sqrt(mean_squared_error(y_test_dbp, y_pred_dbp))
        sbp_rmse = np.sqrt(mean_squared_error(y_test_sbp, y_pred_sbp))
        
        dbp_corr, _ = pearsonr(y_test_dbp.flatten(), y_pred_dbp.flatten())
        sbp_corr, _ = pearsonr(y_test_sbp.flatten(), y_pred_sbp.flatten())
        
        # Calculate ISO standard (% of measurements within 10 mmHg)
        dbp_within_10 = np.mean(np.abs(y_test_dbp - y_pred_dbp) < 10) * 100
        sbp_within_10 = np.mean(np.abs(y_test_sbp - y_pred_sbp) < 10) * 100
        
        return {
            'dbp_rmse': dbp_rmse,
            'sbp_rmse': sbp_rmse,
            'dbp_corr': dbp_corr,
            'sbp_corr': sbp_corr,
            'dbp_within_10': dbp_within_10,
            'sbp_within_10': sbp_within_10
        }
        

class DANNModel:
    """
    Domain-Adversarial Neural Network for blood pressure estimation.
    
    This implements the DANN approach described in the paper for
    minimally-trained personalized blood pressure estimation.
    """
    def __init__(self, input_shape=(9,), lstm_units=64, hidden_size=30, dropout_rate=0.2, 
                 num_domains=3, lambda_param=1.0):
        """
        Initialize DANN model.
        
        Args:
            input_shape: Shape of input features
            lstm_units: Number of LSTM units
            hidden_size: Size of hidden layers in task-specific networks
            dropout_rate: Dropout rate for regularization
            num_domains: Number of domains (subjects) for classification
            lambda_param: Weight for domain classification loss
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_domains = num_domains
        self.lambda_param = lambda_param
        
        # Build model components
        self.feature_extractor, self.bp_estimator, self.domain_classifier = self._build_components()
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Loss functions
        self.bp_loss_fn = tf.keras.losses.MeanSquaredError()
        self.domain_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
    def _build_components(self):
        """
        Build the three components of the DANN model:
        1. Feature extractor
        2. BP estimator
        3. Domain classifier
        """
        # Feature Extractor
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Reshape((self.input_shape[0], 1))(inputs)
        x = tf.keras.layers.LSTM(self.lstm_units)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        shared = tf.keras.layers.Dense(self.hidden_size, activation='relu')(x)
        
        feature_extractor = tf.keras.models.Model(inputs=inputs, outputs=shared, name='feature_extractor')
        
        # BP Estimator
        bp_inputs = tf.keras.layers.Input(shape=(self.hidden_size,))
        
        # DBP estimation
        dbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(bp_inputs)
        dbp = tf.keras.layers.Dropout(self.dropout_rate)(dbp)
        dbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(dbp)
        dbp = tf.keras.layers.Dropout(self.dropout_rate)(dbp)
        dbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(dbp)
        dbp = tf.keras.layers.Dropout(self.dropout_rate)(dbp)
        dbp_output = tf.keras.layers.Dense(1, name='dbp_output')(dbp)
        
        # SBP estimation
        sbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(bp_inputs)
        sbp = tf.keras.layers.Dropout(self.dropout_rate)(sbp)
        sbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(sbp)
        sbp = tf.keras.layers.Dropout(self.dropout_rate)(sbp)
        sbp = tf.keras.layers.Dense(self.hidden_size, activation='relu')(sbp)
        sbp = tf.keras.layers.Dropout(self.dropout_rate)(sbp)
        sbp_output = tf.keras.layers.Dense(1, name='sbp_output')(sbp)
        
        bp_estimator = tf.keras.models.Model(
            inputs=bp_inputs, 
            outputs=[dbp_output, sbp_output],
            name='bp_estimator'
        )
        
        # Domain Classifier
        domain_inputs = tf.keras.layers.Input(shape=(self.hidden_size,))
        
        # Apply gradient reversal (implemented in the custom training loop)
        reversed_features = GradientReversalLayer()(domain_inputs)
        
        # Domain classification layers
        domain = tf.keras.layers.Dense(self.hidden_size, activation='relu')(reversed_features)
        domain = tf.keras.layers.Dropout(self.dropout_rate)(domain)
        domain = tf.keras.layers.Dense(self.hidden_size, activation='relu')(domain)
        domain = tf.keras.layers.Dropout(self.dropout_rate)(domain)
        domain_output = tf.keras.layers.Dense(self.num_domains, name='domain_output')(domain)
        
        domain_classifier = tf.keras.models.Model(
            inputs=domain_inputs,
            outputs=domain_output,
            name='domain_classifier'
        )
        
        return feature_extractor, bp_estimator, domain_classifier
    
    @tf.function
    def train_step(self, x, y_bp, y_domain):
        """
        Custom training step for adversarial training.
        
        Args:
            x: Input features
            y_bp: Blood pressure labels [DBP, SBP]
            y_domain: Domain labels
            
        Returns:
            Tuple of losses (BP loss, domain loss)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            features = self.feature_extractor(x, training=True)
            bp_outputs = self.bp_estimator(features, training=True)
            domain_outputs = self.domain_classifier(features, training=True)
            
            # Calculate losses
            dbp_loss = self.bp_loss_fn(y_bp[:, 0:1], bp_outputs[0])
            sbp_loss = self.bp_loss_fn(y_bp[:, 1:2], bp_outputs[1])
            bp_loss = dbp_loss + sbp_loss
            
            domain_loss = self.domain_loss_fn(y_domain, domain_outputs)
            
            # Total loss with gradient reversal effect
            total_loss = bp_loss - self.lambda_param * domain_loss
        
        # Get gradients
        gradients = tape.gradient(total_loss, 
                                 self.feature_extractor.trainable_variables + 
                                 self.bp_estimator.trainable_variables)
        
        # Apply gradients to feature extractor and BP estimator
        self.optimizer.apply_gradients(zip(
            gradients,
            self.feature_extractor.trainable_variables + self.bp_estimator.trainable_variables
        ))
        
        # Train domain classifier separately (maximizing domain loss)
        with tf.GradientTape() as tape:
            features = self.feature_extractor(x, training=False)  # No gradient flow to feature extractor
            domain_outputs = self.domain_classifier(features, training=True)
            domain_loss = self.domain_loss_fn(y_domain, domain_outputs)
        
        domain_gradients = tape.gradient(domain_loss, self.domain_classifier.trainable_variables)
        self.optimizer.apply_gradients(zip(domain_gradients, self.domain_classifier.trainable_variables))
        
        return bp_loss, domain_loss
    
    def train(self, data, new_subject_id, source_subject_ids, 
              training_minutes=4, epochs=100, batch_size=32):
        """
        Train the DANN model with reduced data from a new subject,
        using knowledge transfer from source subjects.
        
        Args:
            data: Dictionary containing processed data for all subjects
            new_subject_id: ID of the new subject with reduced training data
            source_subject_ids: List of IDs for source and target domains
            training_minutes: Number of minutes of training data to use
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training history
        """
        assert len(source_subject_ids) == 2, "Need exactly 2 source subjects"
        
        # Get data for the new subject
        new_subject_data = data[new_subject_id]
        
        # Calculate number of samples for X minutes
        # Assuming each beat takes about 1 second on average (60 BPM)
        samples_per_minute = 60
        n_samples = training_minutes * samples_per_minute
        
        # Limit training data for new subject
        X_new_train = new_subject_data['train']['X'][:n_samples]
        y_new_train = new_subject_data['train']['y'][:n_samples]
        
        # Get validation and test data for the new subject
        X_new_val = new_subject_data['val']['X']
        y_new_val = new_subject_data['val']['y']
        X_new_test = new_subject_data['test']['X']
        y_new_test = new_subject_data['test']['y']
        
        # Get data for source subjects
        X_source_train = []
        y_source_train = []
        
        for idx, subject_id in enumerate(source_subject_ids):
            source_data = data[subject_id]
            X_source_train.append(source_data['train']['X'])
            y_source_train.append(source_data['train']['y'])
        
        # Combine data from all sources and the new subject
        # Create domain labels (0: new subject, 1: source1, 2: source2)
        X_train_list = [X_new_train] + X_source_train
        y_train_list = [y_new_train] + y_source_train
        domain_labels_list = []
        
        for i, X in enumerate(X_train_list):
            domain_labels_list.append(np.ones(len(X), dtype=np.int32) * i)
        
        # Balance the data to avoid domain classifier issues
        min_samples = min(len(X) for X in X_train_list)
        
        balanced_X_train_list = []
        balanced_y_train_list = []
        balanced_domain_labels_list = []
        
        for i, (X, y, d) in enumerate(zip(X_train_list, y_train_list, domain_labels_list)):
            indices = np.random.choice(len(X), min_samples, replace=False)
            balanced_X_train_list.append(X[indices])
            balanced_y_train_list.append(y[indices])
            balanced_domain_labels_list.append(d[indices])
        
        # Concatenate balanced data
        X_train = np.concatenate(balanced_X_train_list, axis=0)
        y_train = np.concatenate(balanced_y_train_list, axis=0)
        domain_labels = np.concatenate(balanced_domain_labels_list, axis=0)
        
        # Shuffle the data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        domain_labels = domain_labels[shuffle_idx]
        
        # Training loop
        history = {
            'bp_loss': [],
            'domain_loss': [],
            'val_dbp_rmse': [],
            'val_sbp_rmse': []
        }
        
        for epoch in range(epochs):
            # Batch training
            bp_losses = []
            domain_losses = []
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_bp_batch = y_train[i:i+batch_size]
                y_domain_batch = domain_labels[i:i+batch_size]
                
                bp_loss, domain_loss = self.train_step(X_batch, y_bp_batch, y_domain_batch)
                bp_losses.append(bp_loss.numpy())
                domain_losses.append(domain_loss.numpy())
            
            # Calculate mean losses
            mean_bp_loss = np.mean(bp_losses)
            mean_domain_loss = np.mean(domain_losses)
            
            # Validation on new subject
            val_metrics = self.evaluate(X_new_val, y_new_val)
            
            # Log history
            history['bp_loss'].append(mean_bp_loss)
            history['domain_loss'].append(mean_domain_loss)
            history['val_dbp_rmse'].append(val_metrics['dbp_rmse'])
            history['val_sbp_rmse'].append(val_metrics['sbp_rmse'])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - BP Loss: {mean_bp_loss:.4f}, Domain Loss: {mean_domain_loss:.4f}, "
                      f"Val DBP RMSE: {val_metrics['dbp_rmse']:.4f}, Val SBP RMSE: {val_metrics['sbp_rmse']:.4f}")
        
        # After DANN training, retrain the model on the new subject data only
        print("\nRetraining model on new subject data only...")
        for epoch in range(epochs // 2):  # Fewer epochs for fine-tuning
            bp_losses = []
            
            for i in range(0, len(X_new_train), batch_size):
                X_batch = X_new_train[i:i+batch_size]
                y_bp_batch = y_new_train[i:i+batch_size]
                
                with tf.GradientTape() as tape:
                    features = self.feature_extractor(X_batch, training=True)
                    bp_outputs = self.bp_estimator(features, training=True)
                    
                    dbp_loss = self.bp_loss_fn(y_bp_batch[:, 0:1], bp_outputs[0])
                    sbp_loss = self.bp_loss_fn(y_bp_batch[:, 1:2], bp_outputs[1])
                    bp_loss = dbp_loss + sbp_loss
                
                gradients = tape.gradient(
                    bp_loss,
                    self.feature_extractor.trainable_variables + self.bp_estimator.trainable_variables
                )
                
                self.optimizer.apply_gradients(zip(
                    gradients,
                    self.feature_extractor.trainable_variables + self.bp_estimator.trainable_variables
                ))
                
                bp_losses.append(bp_loss.numpy())
            
            if epoch % 10 == 0:
                print(f"Fine-tuning Epoch {epoch}/{epochs//2} - BP Loss: {np.mean(bp_losses):.4f}")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the DANN model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels [DBP, SBP]
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Forward pass
        features = self.feature_extractor(X_test, training=False)
        bp_outputs = self.bp_estimator(features, training=False)
        
        y_pred_dbp = bp_outputs[0].numpy()
        y_pred_sbp = bp_outputs[1].numpy()
        
        y_test_dbp = y_test[:, 0:1]
        y_test_sbp = y_test[:, 1:2]
        
        # Calculate metrics
        dbp_rmse = np.sqrt(mean_squared_error(y_test_dbp, y_pred_dbp))
        sbp_rmse = np.sqrt(mean_squared_error(y_test_sbp, y_pred_sbp))
        
        dbp_corr, _ = pearsonr(y_test_dbp.flatten(), y_pred_dbp.flatten())
        sbp_corr, _ = pearsonr(y_test_sbp.flatten(), y_pred_sbp.flatten())
        
        # Calculate ISO standard (% of measurements within 10 mmHg)
        dbp_within_10 = np.mean(np.abs(y_test_dbp - y_pred_dbp) < 10) * 100
        sbp_within_10 = np.mean(np.abs(y_test_sbp - y_pred_sbp) < 10) * 100
        
        return {
            'dbp_rmse': dbp_rmse,
            'sbp_rmse': sbp_rmse,
            'dbp_corr': dbp_corr,
            'sbp_corr': sbp_corr,
            'dbp_within_10': dbp_within_10,
            'sbp_within_10': sbp_within_10
        }
    
    def predict(self, X):
        """
        Generate blood pressure predictions.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of DBP and SBP predictions
        """
        features = self.feature_extractor(X, training=False)
        bp_outputs = self.bp_estimator(features, training=False)
        
        return bp_outputs[0].numpy(), bp_outputs[1].numpy()
    
    def plot_bland_altman(self, X_test, y_test, title='Bland-Altman Plot'):
        """
        Generate Bland-Altman plot for the model predictions.
        
        Args:
            X_test: Test features
            y_test: Test labels [DBP, SBP]
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Generate predictions
        dbp_pred, sbp_pred = self.predict(X_test)
        
        # Extract true values
        dbp_true = y_test[:, 0:1]
        sbp_true = y_test[:, 1:2]
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # DBP Bland-Altman
        dbp_mean = (dbp_true.flatten() + dbp_pred.flatten()) / 2
        dbp_diff = dbp_true.flatten() - dbp_pred.flatten()
        
        dbp_mean_diff = np.mean(dbp_diff)
        dbp_std_diff = np.std(dbp_diff)
        
        axes[0].scatter(dbp_mean, dbp_diff, alpha=0.5)
        axes[0].axhline(dbp_mean_diff, color='k', linestyle='-', linewidth=1)
        axes[0].axhline(dbp_mean_diff + 1.96 * dbp_std_diff, color='r', linestyle='--', linewidth=1)
        axes[0].axhline(dbp_mean_diff - 1.96 * dbp_std_diff, color='r', linestyle='--', linewidth=1)
        axes[0].set_title('DBP Bland-Altman Plot')
        axes[0].set_xlabel('Mean (mmHg)')
        axes[0].set_ylabel('Difference (mmHg)')
        
        # SBP Bland-Altman
        sbp_mean = (sbp_true.flatten() + sbp_pred.flatten()) / 2
        sbp_diff = sbp_true.flatten() - sbp_pred.flatten()
        
        sbp_mean_diff = np.mean(sbp_diff)
        sbp_std_diff = np.std(sbp_diff)
        
        axes[1].scatter(sbp_mean, sbp_diff, alpha=0.5)
        axes[1].axhline(sbp_mean_diff, color='k', linestyle='-', linewidth=1)
        axes[1].axhline(sbp_mean_diff + 1.96 * sbp_std_diff, color='r', linestyle='--', linewidth=1)
        axes[1].axhline(sbp_mean_diff - 1.96 * sbp_std_diff, color='r', linestyle='--', linewidth=1)
        axes[1].set_title('SBP Bland-Altman Plot')
        axes[1].set_xlabel('Mean (mmHg)')
        axes[1].set_ylabel('Difference (mmHg)')
        
        # Calculate percentage of points within 10 mmHg
        dbp_within_10 = np.mean(np.abs(dbp_diff) < 10) * 100
        sbp_within_10 = np.mean(np.abs(sbp_diff) < 10) * 100
        
        # Add text annotations
        axes[0].text(0.05, 0.95, f"{dbp_within_10:.1f}% within 10 mmHg", 
                     transform=axes[0].transAxes, fontsize=10,
                     verticalalignment='top')
        
        axes[1].text(0.05, 0.95, f"{sbp_within_10:.1f}% within 10 mmHg", 
                     transform=axes[1].transAxes, fontsize=10,
                     verticalalignment='top')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.05)
        
        return fig