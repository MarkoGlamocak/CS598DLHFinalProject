# CS598DLHFinalProject

# Domain-Adversarial Neural Networks for Blood Pressure Estimation

This repository contains the implementation of the paper "Developing Personalized Models of Blood Pressure Estimation from Wearable Sensors Data Using Minimally-trained Domain Adversarial Neural Networks" by Zhang et al.

## Project Overview

This implementation focuses on applying Domain-Adversarial Neural Networks (DANN) to enable personalized blood pressure estimation with minimal training data. The key challenge addressed is reducing the amount of calibration data required from a new user while maintaining clinical-grade accuracy (within ISO standards).

## Key Components

1. **Data Generation and Preprocessing**:
   - Synthetic dataset generator for bioimpedance signals
   - Data preprocessing pipeline simulating the paper's approach

2. **Base MTL BP Estimation Model**:
   - LSTM-based feature extractor
   - Shared dense layer
   - Dual task-specific networks for diastolic and systolic blood pressure

3. **Domain-Adversarial Neural Network (DANN)**:
   - Feature extractor (shared with the base model)
   - BP estimator
   - Domain classifier with gradient reversal
   - Implementation of adversarial training methodology

4. **Experiment Framework**:
   - Testing with different amounts of training data (3, 4, and 5 minutes)
   - Comparison between baseline, pretrained, and DANN approaches
   - Evaluation using RMSE, correlation, and ISO standard compliance

## Repository Structure

```
.
├── data/                      # Directory for datasets
├── results/                   # Directory for experimental results
├── main.py                    # Data preprocessing module
├── dann_bp_model.py           # Implementation of BP estimation models
├── experiment_runner.py       # Experiment execution and evaluation
├── bioimpedance_data_generator.py  # Synthetic dataset generator
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- SciPy
- scikit-learn

### Installation

```bash
# Clone the repository
git clone https://github.com/MarkoGlamocak/cufflessbp_dann.git
cd cufflessbp_dann

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

1. **Generate synthetic dataset**:
   ```bash
   python bioimpedance_data_generator.py
   ```

2. **Run experiments**:
   ```bash
   python experiment_runner.py
   ```

## Model Architecture

### Base MTL Model

The base model consists of:
- An LSTM layer to process the 9-dimensional input sequences
- A shared dense layer to extract features
- Two task-specific networks for diastolic and systolic blood pressure
- Dropout layers for regularization

### DANN Model

The DANN approach extends the base model with:
- A domain classifier that tries to predict which subject a beat belongs to
- A gradient reversal layer between the feature extractor and domain classifier
- Adversarial training mechanism to force the feature extractor to learn subject-invariant features

## Training Approach

The adversarial training follows the algorithm defined in the paper:

- θBP = θBP + α · ∂LBP/∂θBP
- θd = θd + α · λ · ∂Ld/∂θd
- θf = θf + α · (-λ · ∂Ld/∂θf + ∂LBP/∂θf)

Where:
- θBP, θd, θf are parameters for BP estimator, domain classifier, and feature extractor
- LBP is the loss for BP estimation: ∑(EiS - TiS)² + (EiD - TiD)²
- Ld is the cross-entropy loss for domain classification
- α is the learning rate
- λ is the loss weight balancing BP estimator and domain classifier

## Experimental Results

The experiments test the DANN model with different amounts of training data (3, 4, and 5 minutes) and compare it with:
1. Directly training the base model with limited data
2. Using a pretrained model from another subject

Key metrics include:
- Root Mean Square Error (RMSE) in mmHg
- Correlation coefficient (R)
- ISO standard compliance (percentage of measurements within 10 mmHg)

## Acknowledgments

This project is based on the following paper:
```
Zhang, L., Hurley, N. C., Ibrahim, B., Spatz, E., Krumholz, H. M., Jafari, R., & Mortazavi, B. J. (2020). 
Developing Personalized Models of Blood Pressure Estimation from Wearable Sensors Data Using 
Minimally-trained Domain Adversarial Neural Networks. Proceedings of Machine Learning Research, 126, 97-120.
```

## License

This project is licensed under the MIT License.
