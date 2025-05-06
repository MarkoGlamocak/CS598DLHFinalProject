# Domain-Adversarial Neural Network for Blood Pressure Estimation

## Overview

This project implements a Domain-Adversarial Neural Network (DANN) architecture for estimating both systolic blood pressure (SBP) and diastolic blood pressure (DBP) from bioimpedance signals measured at the wrist. The implementation addresses the challenge of cross-subject variability in bioimpedance signals by using a domain adaptation approach.

The key innovation is the use of a gradient reversal layer that allows the feature extractor to learn domain-invariant features, making the model more generalizable across different subjects with limited data.

## Repository Structure

- `flip_gradient.py`: Implementation of the Gradient Reversal Layer for domain adaptation
- `main_dann.py`: Main script containing the DANN model implementation and experiment runner
- `data_preparation.py`: Data preprocessing functions to prepare bioimpedance data
- `Bioimpedance_BP_Data_Generator.py`: Script to generate synthetic bioimpedance data for experimentation

## Requirements

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn

## Usage

### Data Generation

To generate synthetic bioimpedance data for experimentation:

```bash
python Bioimpedance_BP_Data_Generator.py
```

This will create a `data` directory with synthetic bioimpedance signals for 11 subjects.

### Running Experiments

To run the DANN experiments:

```bash
python main_dann.py
```

The script will:
1. Preprocess the data
2. Run experiments with different training durations (3, 4, and 5 minutes)
3. Generate Bland-Altman plots for evaluation
4. Save results summary to `dann_results_summary.txt`

### Custom Experiments

To customize the experiments, modify the parameters in the `main()` function in `main_dann.py`:

```python
# Test different durations of training data
test_durations = [3, 4, 5]  # minutes

# Number of runs for robustness
num_runs = 10
```

## Model Architecture

The DANN architecture consists of three main components:

1. **Feature Extractor**: LSTM-based feature extractor that learns domain-invariant features
2. **BP Estimators**: Two branches for estimating DBP and SBP values
3. **Domain Classifier**: Identifies the domain (subject) of the input data, with a gradient reversal layer to encourage domain-invariant features

## Evaluation Metrics

The model is evaluated using several metrics:

- **Root Mean Square Error (RMSE)**: Measures the average prediction error
- **Correlation Coefficient**: Assesses the linear relationship between predicted and reference values
- **Percentage Within 10 mmHg**: Calculates the percentage of estimations within 10 mmHg of the reference value
- **Bland-Altman Plots**: Visualizes the agreement between the estimation and reference methods

## Results

The system generates a comprehensive results summary in `dann_results_summary.txt`, which includes:

- Average results across all subjects for each training duration (3, 4, and 5 minutes)
- Detailed metrics for each subject and training duration

Bland-Altman plots are saved in the `bland_altman_plots` directory.

## References

- Zhang, L., Hurley, N. C., Ibrahim, B., Spatz, E., Krumholz,
H. M., Jafari, R., & Mortazavi, B. J. (2020). Developing Per-
sonalized Models of Blood Pressure Estimation from Wear-
able Sensors Data Using Minimally-trained Domain Adver-
sarial Neural Networks. Proceedings of Machine Learning
Research, 126, 97-120.
