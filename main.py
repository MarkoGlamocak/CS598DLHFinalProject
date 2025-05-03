import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def data_preprocess():
    """
    Read and preprocess the bioimpedance BP dataset:
    1. Load the dataset from CSV
    2. Normalize feature columns to [0, 1] range
    3. Split data by subject ID into train/validation/test sets (80%/10%/10%)
    
    Returns:
        dict: Dictionary with keys as subject IDs, each containing dicts with train/val/test splits
    """
    
    # Define feature columns and target columns
    feature_columns = [
        'Channel_1', 'Channel_2', 'Channel_3', 'Channel_4',
        'Channel_1_Derivative', 'Channel_2_Derivative', 
        'Channel_3_Derivative', 'Channel_4_Derivative',
        'Timing'
    ]
    target_columns = ['DBP', 'SBP']
    
    # Read the dataset
    df = pd.read_csv('bioimpedance_bp_dataset.csv')
    
    # Initialize the result dictionary
    subjects_data = {}
    
    # Initialize scalers for features - one for each feature to preserve individual characteristics
    feature_scalers = {}
    for col in feature_columns:
        feature_scalers[col] = MinMaxScaler()
    
    # Initialize target scalers
    target_scalers = {}
    for col in target_columns:
        target_scalers[col] = MinMaxScaler()
    
    # Get all unique subject IDs
    subject_ids = df['Subject_ID'].unique()
    
    # Process each subject independently
    for subject_id in subject_ids:
        # Get data for this subject
        subject_data = df[df['Subject_ID'] == subject_id]
        
        # Create copies for processing
        subject_features = subject_data[feature_columns].copy()
        subject_targets = subject_data[target_columns].copy()
        
        # Normalize features
        normalized_features = subject_features.copy()
        for col in feature_columns:
            # Fit transform on each feature independently for this subject
            normalized_features[col] = feature_scalers[col].fit_transform(
                subject_features[[col]]
            ).ravel()
        
        # Normalize targets (optional - depends on use case)
        normalized_targets = subject_targets.copy()
        for col in target_columns:
            normalized_targets[col] = target_scalers[col].fit_transform(
                subject_targets[[col]]
            ).ravel()
        
        # Split into train/validation/test sets
        n_samples = len(subject_data)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        
        # Create indices for splitting
        indices = np.arange(n_samples)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Create splits
        train_data = {
            'features': normalized_features.iloc[train_idx],
            'targets': normalized_targets.iloc[train_idx],
            'activity': subject_data['Activity'].iloc[train_idx],
            'scalers': {
                'features': {col: feature_scalers[col] for col in feature_columns},
                'targets': {col: target_scalers[col] for col in target_columns}
            }
        }
        
        val_data = {
            'features': normalized_features.iloc[val_idx],
            'targets': normalized_targets.iloc[val_idx],
            'activity': subject_data['Activity'].iloc[val_idx]
        }
        
        test_data = {
            'features': normalized_features.iloc[test_idx],
            'targets': normalized_targets.iloc[test_idx],
            'activity': subject_data['Activity'].iloc[test_idx]
        }
        
        # Store in result dictionary
        subjects_data[subject_id] = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    return subjects_data

# Example usage
if __name__ == "__main__":
    # Process the data
    processed_data = data_preprocess()
    
    # Display information about processed data
    print(f"Processed data for {len(processed_data)} subjects")
    
    # Show example for subject 1
    subject_1_data = processed_data[1]
    print(f"\nSubject 1 data splits:")
    print(f"Training samples: {len(subject_1_data['train']['features'])}")
    print(f"Validation samples: {len(subject_1_data['validation']['features'])}")
    print(f"Test samples: {len(subject_1_data['test']['features'])}")
    
    # Show normalized feature ranges
    print("\nFeature ranges after normalization (Subject 1, training data):")
    for col in subject_1_data['train']['features'].columns:
        min_val = subject_1_data['train']['features'][col].min()
        max_val = subject_1_data['train']['features'][col].max()
        print(f"{col}: [{min_val:.4f}, {max_val:.4f}]")
