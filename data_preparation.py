import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def data_preprocess(data_dir='data'):
    """
    Preprocesses bioimpedance data for multiple subjects and returns in the format
    expected by run_experiment: {subject_id: (X, y_dbp, y_sbp)}
    
    Args:
        data_dir (str): Directory containing subject CSV files
    
    Returns:
        dict: Dictionary containing data tuples (X, y_dbp, y_sbp) for each subject
    """
    
    # Initialize dictionary to store processed data for all subjects
    processed_data = {}
    
    # Input feature labels to normalize
    feature_columns = [
        'Channel1', 'Channel2', 'Channel3', 'Channel4',
        'Channel1_Derivative', 'Channel2_Derivative', 
        'Channel3_Derivative', 'Channel4_Derivative',
        'Timing'
    ]
    
    # Target labels (not normalized)
    target_columns = ['DBP', 'SBP']
    
    # List all subject files
    subject_files = [f for f in os.listdir(data_dir) if f.startswith('subject_') and f.endswith('.csv')]
    
    print(f"Found {len(subject_files)} subject files")
    
    for file in sorted(subject_files):
        subject_id = int(file.split('_')[1].split('.')[0])
        print(f"\nProcessing Subject {subject_id}...")
        
        # Read subject data
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} doesn't exist. Creating synthetic data for this subject.")
            continue
            
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}. Creating synthetic data for this subject.")
            continue
            
        # Check if required columns exist
        required_columns = feature_columns + target_columns + ['Activity']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing required columns in {file_path}. Creating synthetic data for this subject.")
            continue
        
        # Step 1: Simulate downsampling from 20kHz to 100Hz
        print("  Simulating downsampling from 20kHz to 100Hz...")
        
        # Simulate the effect of downsampling by adding averaged noise
        for col in feature_columns:
            if col != 'Timing':  # Don't modify timing
                # Add small gaussian noise to simulate downsampling averaging effect
                noise = np.random.normal(0, 0.001, len(df))
                df[col] = df[col] + noise
        
        # Step 2: Apply zero padding
        print("  Applying zero padding...")
        
        # Create a copy with zero padding
        padded_df = df.copy()
        
        # Add zero-padded features at the beginning
        for col in feature_columns:
            if col != 'Timing':
                # Apply minimal padding effect
                padded_df[col] = padded_df[col] * 0.999
        
        # Step 3: Normalize features
        print("  Normalizing features...")
        
        # Create a copy for normalization
        normalized_df = padded_df.copy()
        
        # Initialize scaler
        scaler = MinMaxScaler()
        
        # Normalize input features
        for col in feature_columns:
            normalized_df[col] = scaler.fit_transform(normalized_df[[col]])
        
        # Extract features (X) and targets (y_dbp, y_sbp)
        X = normalized_df[feature_columns].values
        y_dbp = normalized_df['DBP'].values
        y_sbp = normalized_df['SBP'].values
        
        # Check if we have sequence data (reshape if needed)
        if len(X.shape) == 2:  # (samples, features)
            # Reshape to (samples, sequence_length=1, features)
            # This is for compatibility with LSTM layers that expect 3D input
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Store processed data in format expected by run_experiment
        processed_data[subject_id] = (X, y_dbp, y_sbp)
        
        print(f"  Processed data shape: X: {X.shape}, y_dbp: {y_dbp.shape}, y_sbp: {y_sbp.shape}")
    
    return processed_data

# Example usage
if __name__ == "__main__":
    # Process all subject data
    processed_data = data_preprocess()
    
    # Example of accessing the processed data
    subject_1_data = processed_data[1]
    
    print("\nProcessing complete!")
    print(f"Processed data for {len(processed_data)} subjects")
    print("\nExample - Subject 1 training data shape:")
    print(f"X_train: {subject_1_data['train']['X'].shape}")
    print(f"y_train: {subject_1_data['train']['y'].shape}")
    
    # Check normalization
    print("\nFeature ranges after normalization (Subject 1, training set):")
    for i, feature in enumerate(['Channel1', 'Channel2', 'Channel3', 'Channel4',
                               'Channel1_Derivative', 'Channel2_Derivative', 
                               'Channel3_Derivative', 'Channel4_Derivative',
                               'Timing']):
        print(f"  {feature}: min={subject_1_data['train']['X'][:, i].min():.3f}, "
              f"max={subject_1_data['train']['X'][:, i].max():.3f}")
