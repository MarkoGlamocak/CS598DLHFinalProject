import numpy as np
import pandas as pd
from scipy import signal
import random

class BioimpedanceBPDataGenerator:
    def __init__(self, n_subjects=11, n_beats_per_subject=2000):
        self.n_subjects = n_subjects
        self.n_beats_per_subject = n_beats_per_subject
        self.sampling_rate = 100  # 100 Hz as mentioned in paper
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Subject-specific baseline variations
        self.subject_variations = self._generate_subject_variations()
        
    def _generate_subject_variations(self):
        """Generate subject-specific baseline variations"""
        variations = {}
        for subject_id in range(1, self.n_subjects + 1):
            variations[subject_id] = {
                'dbp_baseline': np.random.uniform(65, 75),  # Baseline DBP varies by subject
                'sbp_baseline': np.random.uniform(105, 115),  # Baseline SBP varies by subject
                'impedance_baseline': np.random.uniform(80, 120),  # Baseline impedance
                'amplitude_scaling': np.random.uniform(0.8, 1.2),  # Overall amplitude scaling
                'noise_level': np.random.uniform(0.02, 0.05)  # Subject-specific noise
            }
        return variations
    
    def _generate_activity_sequence(self):
        """Generate realistic activity sequence"""
        activities = []
        bp_values = []
        
        # Define activity parameters
        activity_params = {
            'rest': {
                'duration_range': (300, 500),  # beats
                'dbp_range': (60, 75),
                'sbp_range': (100, 120),
                'transition_rate': 'slow'
            },
            'light_exercise': {
                'duration_range': (200, 400),
                'dbp_range': (65, 85),
                'sbp_range': (115, 135),
                'transition_rate': 'medium'
            },
            'intense_exercise': {
                'duration_range': (100, 200),
                'dbp_range': (75, 90),
                'sbp_range': (130, 150),
                'transition_rate': 'fast'
            },
            'recovery': {
                'duration_range': (150, 300),
                'dbp_range': (70, 85),
                'sbp_range': (110, 130),
                'transition_rate': 'slow'
            }
        }
        
        current_beat = 0
        current_dbp = 67.5  # Starting values
        current_sbp = 110
        
        # Generate activity sequence
        activity_sequence = ['rest', 'light_exercise', 'intense_exercise', 'recovery']
        
        while current_beat < self.n_beats_per_subject:
            for activity in activity_sequence:
                if current_beat >= self.n_beats_per_subject:
                    break
                    
                params = activity_params[activity]
                duration = random.randint(*params['duration_range'])
                target_dbp = random.uniform(*params['dbp_range'])
                target_sbp = random.uniform(*params['sbp_range'])
                
                # Generate transition with realistic smoothing
                for i in range(min(duration, self.n_beats_per_subject - current_beat)):
                    # Transition rate based on activity
                    if params['transition_rate'] == 'slow':
                        alpha = 0.02
                    elif params['transition_rate'] == 'medium':
                        alpha = 0.05
                    else:  # fast
                        alpha = 0.1
                    
                    # Smooth transition to target BP
                    current_dbp = current_dbp * (1 - alpha) + target_dbp * alpha
                    current_sbp = current_sbp * (1 - alpha) + target_sbp * alpha
                    
                    # Add physiological variation
                    dbp_variation = np.random.normal(0, 1.5)
                    sbp_variation = np.random.normal(0, 2.5)
                    
                    final_dbp = current_dbp + dbp_variation
                    final_sbp = current_sbp + sbp_variation
                    
                    activities.append(activity)
                    bp_values.append((final_dbp, final_sbp))
                    current_beat += 1
        
        return activities, bp_values
    
    def _generate_bioimpedance_signal(self, subject_id, activity, dbp, sbp):
        """Generate realistic bioimpedance signal for one beat"""
        var = self.subject_variations[subject_id]
        
        # Signal length for one heartbeat (~1 second at 100 Hz)
        signal_length = 100
        t = np.linspace(0, 1, signal_length)
        
        # Base impedance waveform (simplified representation)
        base_impedance = var['impedance_baseline'] * np.ones(signal_length)
        
        # Add characteristic peaks based on BP
        # Main systolic peak
        peak_amp = (sbp - var['sbp_baseline']) / 50 * var['amplitude_scaling']
        peak_time = 0.3  # Approximate timing of systolic peak
        systolic_peak = peak_amp * np.exp(-((t - peak_time) ** 2) / (2 * 0.05 ** 2))
        
        # Diastolic component (smaller)
        diastolic_amp = (dbp - var['dbp_baseline']) / 50 * var['amplitude_scaling'] * 0.3
        diastolic_time = 0.45
        diastolic_peak = diastolic_amp * np.exp(-((t - diastolic_time) ** 2) / (2 * 0.08 ** 2))
        
        # Combine components
        impedance_signal = base_impedance + systolic_peak + diastolic_peak
        
        # Add activity-dependent noise
        if activity == 'intense_exercise':
            noise_factor = 1.5
        elif activity == 'light_exercise':
            noise_factor = 1.2
        else:
            noise_factor = 1.0
            
        noise = np.random.normal(0, var['noise_level'] * noise_factor, signal_length)
        impedance_signal += noise
        
        # Apply signal artifacts based on activity
        if activity in ['light_exercise', 'intense_exercise']:
            motion_artifact = 0.3 * np.sin(2 * np.pi * 0.5 * t) * (activity == 'intense_exercise' and 2 or 1)
            impedance_signal += motion_artifact
        
        return impedance_signal
    
    def _extract_characteristic_points(self, impedance_signal):
        """Extract characteristic points from impedance signal"""
        # Find peaks and important points
        diastolic_idx = np.argmin(impedance_signal[:30])  # Early diastolic minimum
        systolic_idx = np.argmax(impedance_signal[20:70]) + 20  # Main systolic peak
        
        # Find inflection point (approximation)
        derivative = np.diff(impedance_signal)
        inflection_idx = np.argmax(derivative[:40])  # Before systolic peak
        
        # Find foot point
        foot_idx = np.argmin(impedance_signal[:inflection_idx])
        
        return {
            'diastolic_peak': impedance_signal[diastolic_idx],
            'maximum_slope': derivative[inflection_idx],
            'systolic_foot': impedance_signal[foot_idx],
            'inflection_point': impedance_signal[inflection_idx]
        }
    
    def generate_four_channel_measurements(self, impedance_signal):
        """Generate measurements for 4 bioimpedance channels (ulnar and radial arteries)"""
        channels = []
        
        # Generate 4 channels with slight variations (representing different electrode positions)
        for i in range(4):
            # Add channel-specific variation
            channel_var = 1 + np.random.normal(0, 0.05)
            # Add slight delay to simulate different measurement locations
            delay = int(np.random.uniform(-3, 3))
            
            if delay < 0:
                # Create a copy of the signal, apply roll, and then overwrite the affected portion
                channel_signal = np.roll(impedance_signal.copy(), delay)
                # Fill the wrapped portion with the unwrapped data
                channel_signal[:delay] = impedance_signal[-delay:]
                channel_signal = channel_signal * channel_var
            elif delay > 0:
                # Create a copy of the signal, apply roll, and then overwrite the affected portion
                channel_signal = np.roll(impedance_signal.copy(), delay)
                # Fill the wrapped portion with the unwrapped data
                channel_signal[:delay] = impedance_signal[:delay]
                channel_signal = channel_signal * channel_var
            else:
                # No delay, just apply scaling
                channel_signal = impedance_signal.copy() * channel_var
            
            channels.append(channel_signal)
        
        return channels
    
    def generate_dataset(self):
        """Generate complete synthetic dataset"""
        all_data = []
        
        for subject_id in range(1, self.n_subjects + 1):
            print(f"Generating data for Subject {subject_id}...")
            
            # Generate activity sequence and BP values
            activities, bp_values = self._generate_activity_sequence()
            
            for beat_num in range(self.n_beats_per_subject):
                activity = activities[beat_num]
                dbp, sbp = bp_values[beat_num]
                
                # Generate bioimpedance signal
                impedance_signal = self._generate_bioimpedance_signal(
                    subject_id, activity, dbp, sbp
                )
                
                # Generate 4 channel measurements
                channels = self.generate_four_channel_measurements(impedance_signal)
                
                # Calculate derivatives and timing
                derivatives = [np.diff(channel) for channel in channels]
                timing = np.linspace(0, 1, 100)  # 1 second per beat
                
                # Extract representative values for CSV (using middle of signal)
                mid_idx = 50  # Middle of 100-sample signal
                
                row_data = {
                    'Subject_ID': subject_id,
                    'Beat_Number': beat_num + 1,
                    'Activity': activity,
                    'Channel_1': channels[0][mid_idx],
                    'Channel_2': channels[1][mid_idx],
                    'Channel_3': channels[2][mid_idx],
                    'Channel_4': channels[3][mid_idx],
                    'Channel_1_Derivative': derivatives[0][mid_idx] if mid_idx < len(derivatives[0]) else derivatives[0][-1],
                    'Channel_2_Derivative': derivatives[1][mid_idx] if mid_idx < len(derivatives[1]) else derivatives[1][-1],
                    'Channel_3_Derivative': derivatives[2][mid_idx] if mid_idx < len(derivatives[2]) else derivatives[2][-1],
                    'Channel_4_Derivative': derivatives[3][mid_idx] if mid_idx < len(derivatives[3]) else derivatives[3][-1],
                    'Timing': timing[mid_idx],
                    'DBP': dbp,
                    'SBP': sbp
                }
                
                all_data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        return df

# Generate the dataset
print("Starting dataset generation...")
generator = BioimpedanceBPDataGenerator(n_subjects=11, n_beats_per_subject=2000)
dataset = generator.generate_dataset()

# Save to CSV
output_file = 'bioimpedance_bp_dataset.csv'
dataset.to_csv(output_file, index=False)

print(f"\nDataset generated successfully!")
print(f"Total records: {len(dataset)}")
print(f"Saved to: {output_file}")
print("\nDataset preview:")
print(dataset.head())
print("\nSummary statistics:")
print(dataset.describe())
