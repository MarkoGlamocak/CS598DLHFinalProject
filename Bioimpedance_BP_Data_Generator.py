import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
from scipy.interpolate import interp1d

# Create data directory if it doesn't exist
Path("data").mkdir(exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define physiological parameters for different activities
ACTIVITY_PARAMS = {
    'rest': {
        'dbp_range': (60, 75),
        'sbp_range': (100, 120),
        'heart_rate': (65, 75),
        'bioimpedance_amplitude': (0.8, 1.2),
        'noise_level': 0.02
    },
    'light_exercise': {
        'dbp_range': (65, 85),
        'sbp_range': (115, 135),
        'heart_rate': (75, 95),
        'bioimpedance_amplitude': (0.9, 1.4),
        'noise_level': 0.03
    },
    'intense_exercise': {
        'dbp_range': (75, 90),
        'sbp_range': (130, 150),
        'heart_rate': (95, 120),
        'bioimpedance_amplitude': (1.0, 1.8),
        'noise_level': 0.04
    },
    'recovery': {
        'dbp_range': (68, 82),
        'sbp_range': (110, 130),
        'heart_rate': (80, 90),
        'bioimpedance_amplitude': (0.85, 1.3),
        'noise_level': 0.025
    }
}

def generate_bioimpedance_waveform(time_points, amplitude, phase_shift=0):
    """Generate a realistic bioimpedance waveform for one heartbeat"""
    # Create a characteristic bioimpedance curve with multiple phases
    t = time_points
    
    # Main waveform - combination of peaks representing arterial pulsation
    wave = (
        0.7 * amplitude * np.exp(-((t - (0.2 + phase_shift)) ** 2) / (2 * 0.02 ** 2)) +  # Diastolic peak
        amplitude * np.exp(-((t - (0.4 + phase_shift)) ** 2) / (2 * 0.03 ** 2)) +      # Systolic peak
        0.3 * amplitude * np.exp(-((t - (0.5 + phase_shift)) ** 2) / (2 * 0.02 ** 2)) +  # Dicrotic notch
        0.2 * amplitude * np.exp(-((t - (0.7 + phase_shift)) ** 2) / (2 * 0.04 ** 2))   # Diastolic component
    )
    
    # Add baseline impedance
    baseline = 0.5 + 0.1 * np.sin(2 * np.pi * t * 0.5)
    
    return wave + baseline

def generate_subject_data(subject_id, beats_per_activity=500):
    """Generate bioimpedance data for a single subject"""
    # Subject-specific characteristics
    subject_baseline_dbp = np.random.uniform(65, 75)
    subject_baseline_sbp = np.random.uniform(110, 125)
    subject_impedance_baseline = np.random.uniform(0.4, 0.6)
    subject_artery_delay = np.random.uniform(0.01, 0.03)  # Phase delay between arteries
    
    all_data = []
    beat_counter = 0
    
    # Generate data for each activity type
    for activity, params in ACTIVITY_PARAMS.items():
        for _ in range(beats_per_activity):
            beat_counter += 1
            
            # Generate blood pressure values with subject-specific baseline
            dbp = np.clip(np.random.normal(
                (params['dbp_range'][0] + params['dbp_range'][1]) / 2,
                (params['dbp_range'][1] - params['dbp_range'][0]) / 6
            ) + (subject_baseline_dbp - 70), params['dbp_range'][0], params['dbp_range'][1])
            
            sbp = np.clip(np.random.normal(
                (params['sbp_range'][0] + params['sbp_range'][1]) / 2,
                (params['sbp_range'][1] - params['sbp_range'][0]) / 6
            ) + (subject_baseline_sbp - 117.5), params['sbp_range'][0], params['sbp_range'][1])
            
            # Ensure physiological constraint: SBP > DBP
            if sbp <= dbp:
                sbp = dbp + np.random.uniform(20, 40)
            
            # Generate heart rate
            heart_rate = np.random.uniform(*params['heart_rate'])
            beat_duration = 60 / heart_rate
            
            # Generate timing information
            timing = np.random.uniform(0, 1)  # Beat timing within cardiac cycle
            
            # Generate bioimpedance signals for 4 channels
            time_points = np.linspace(0, 1, 1000)  # High resolution for realistic signals
            amplitudes = np.random.uniform(*params['bioimpedance_amplitude'], size=4)
            
            # Channel 1: Ulnar artery, proximal
            channel1 = generate_bioimpedance_waveform(time_points, amplitudes[0] * subject_impedance_baseline)
            
            # Channel 2: Radial artery, proximal  
            channel2 = generate_bioimpedance_waveform(time_points, amplitudes[1] * subject_impedance_baseline, 
                                                     phase_shift=subject_artery_delay)
            
            # Channel 3: Ulnar artery, distal
            channel3 = generate_bioimpedance_waveform(time_points, amplitudes[2] * subject_impedance_baseline,
                                                     phase_shift=np.random.uniform(0.05, 0.1))
            
            # Channel 4: Radial artery, distal
            channel4 = generate_bioimpedance_waveform(time_points, amplitudes[3] * subject_impedance_baseline,
                                                     phase_shift=subject_artery_delay + np.random.uniform(0.05, 0.1))
            
            # Add realistic noise
            noise = np.random.normal(0, params['noise_level'], size=(4, 1000))
            channels = [channel1, channel2, channel3, channel4]
            
            for i in range(4):
                channels[i] += noise[i]
            
            # Take one point from each channel to match the original data structure
            sample_idx = int(timing * 999)  # Convert timing to sample index
            channel_values = [channel[sample_idx] for channel in channels]
            
            # Calculate derivatives at the sample point
            derivatives = []
            for channel in channels:
                if sample_idx > 0 and sample_idx < 999:
                    derivative = (channel[sample_idx + 1] - channel[sample_idx - 1]) / 2
                else:
                    derivative = channel[1] - channel[0] if sample_idx == 0 else channel[-1] - channel[-2]
                derivatives.append(derivative)
            
            # Store data
            data_row = {
                'Subject_ID': subject_id,
                'Beat_Number': beat_counter,
                'Activity': activity,
                'Channel1': channel_values[0],
                'Channel2': channel_values[1],
                'Channel3': channel_values[2],
                'Channel4': channel_values[3],
                'Channel1_Derivative': derivatives[0],
                'Channel2_Derivative': derivatives[1],
                'Channel3_Derivative': derivatives[2],
                'Channel4_Derivative': derivatives[3],
                'Timing': timing,
                'DBP': dbp,
                'SBP': sbp
            }
            
            all_data.append(data_row)
    
    return pd.DataFrame(all_data)

# Generate data for all 11 subjects
print("Generating bioimpedance dataset for 11 subjects...")
for subject_id in range(1, 12):
    print(f"Generating data for Subject {subject_id}...")
    
    # Generate data
    subject_data = generate_subject_data(subject_id)
    
    # Save to CSV
    filename = f"data/subject_{subject_id:02d}.csv"
    subject_data.to_csv(filename, index=False)
    
    print(f"Saved {len(subject_data)} beats to {filename}")

print("\nDataset generation complete!")
print(f"Generated files in 'data' directory:")
for file in sorted(os.listdir("data")):
    print(f"  - {file}")

# Generate summary statistics
summary_data = []
for subject_id in range(1, 12):
    df = pd.read_csv(f"data/subject_{subject_id:02d}.csv")
    summary = {
        'Subject_ID': subject_id,
        'Total_Beats': len(df),
        'DBP_Mean': df['DBP'].mean(),
        'DBP_Std': df['DBP'].std(),
        'SBP_Mean': df['SBP'].mean(),
        'SBP_Std': df['SBP'].std(),
        'Activities': df['Activity'].value_counts().to_dict()
    }
    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("data/dataset_summary.csv", index=False)

print("\nDataset Summary:")
print(f"Total subjects: 11")
print(f"Total beats per subject: ~2000")
print(f"Activities included: rest, light_exercise, intense_exercise, recovery")
print(f"Blood pressure ranges preserved across activities")
print(f"Subject-specific variations implemented")
print("\nDataset ready for training!")
