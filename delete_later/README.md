# Blood Pressure Estimation using Domain-Adversarial Neural Networks (DANN)

This repository implements the approach described in the paper "Developing Personalized Models of Blood Pressure Estimation from Wearable Sensors Data Using Minimally-trained Domain Adversarial Neural Networks" using synthetic bioimpedance data.

## Overview

This project demonstrates how to use Domain-Adversarial Neural Networks (DANN) to create personalized blood pressure estimation models with minimal training data. The DANN approach allows for knowledge transfer between subjects, enabling accurate blood pressure estimation with as little as 3-5 minutes of training data from a new subject.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Installation

1. Clone this repository:
```bash
git clone https://github.com/MarkoGlamocak/CS598DLHFinalProject.git
cd CS598DLHFinalProject
```

2. Install the required packages:
```bash
pip3 install tensorflow numpy pandas matplotlib scikit-learn
```

## Data Preparation

1. Place your bioimpedance dataset in a directory named `data/` in the root of the project. You can use `Bioimpedance_BP_Data_Generator.py` to generate synthetic data by running the following command:
```bash
python3 Bioimpedance_BP_Data_Generator.py
```
   
2. The data should be structured as CSV files named according to the format `subject_XX.csv` (where XX is the subject ID, e.g., `subject_01.csv`, `subject_02.csv`, etc.).

3. Each CSV file should contain the following columns:
   - `Subject_ID`: Integer
   - `Beat_Number`: Integer
   - `Activity`: String (category of physical activity)
   - `Channel1`: Float (bioimpedance channel 1)
   - `Channel2`: Float (bioimpedance channel 2)
   - `Channel3`: Float (bioimpedance channel 3)
   - `Channel4`: Float (bioimpedance channel 4)
   - `Channel1_Derivative`: Float
   - `Channel2_Derivative`: Float
   - `Channel3_Derivative`: Float
   - `Channel4_Derivative`: Float
   - `Timing`: Float
   - `DBP`: Float (diastolic blood pressure)
   - `SBP`: Float (systolic blood pressure)

## Running the Code

1. Make sure that the `flip_gradient.py` and `bp_dann.py` files are in the root directory of the project.

2. Run the training script:
```bash
python3 run_training.py
```

This script will:
- Train three separate models using 3, 4, and 5 minutes of training data
- Run each training for 100 epochs
- Save the trained models and results

## File Structure

- `Bioimpedance_BP_Data_Generator.py`: Generates synthetic bioimpedance dataset
- `flip_gradient.py`: Implements the gradient reversal layer for domain adaptation
- `bp_dann.py`: Main implementation of the DANN approach for blood pressure estimation
- `run_training.py`: Script to run training with different amounts of data
- `data/`: Directory containing subject CSV files
- `checkpoints/`: Directory where trained models are saved
- `results/`: Directory where evaluation results and plots are saved

## Understanding the Results

After training completes, the script will display a summary of results for each training duration:

```
SUMMARY OF RESULTS
================================================================================
Minutes     DBP RMSE      SBP RMSE      DBP r      SBP r      DBP %      SBP %    
--------------------------------------------------------------------------------
3           x.xx          x.xx          x.xx       x.xx       xx.x       xx.x     
4           x.xx          x.xx          x.xx       x.xx       xx.x       xx.x     
5           x.xx          x.xx          x.xx       x.xx       xx.x       xx.x     
```

Where:
- `DBP RMSE`: Root mean square error for diastolic blood pressure (mmHg)
- `SBP RMSE`: Root mean square error for systolic blood pressure (mmHg)
- `DBP r`: Correlation coefficient for diastolic blood pressure
- `SBP r`: Correlation coefficient for systolic blood pressure
- `DBP %`: Percentage of diastolic blood pressure estimates within 10 mmHg of reference
- `SBP %`: Percentage of systolic blood pressure estimates within 10 mmHg of reference

The ISO standard requires at least 85% of measurements to be within 10 mmHg of the reference. 
Please note the code doesn't currently meet this for SBP.

## Visualizations

The training process generates several visualizations in the `results/` directory:

1. Bland-Altman plots showing the agreement between predicted and reference blood pressure values
2. Scatter plots showing the correlation between predicted and reference blood pressure values

These plots are useful for visually assessing the performance of the models.

## How DANN Works

Domain-Adversarial Neural Networks (DANN) use adversarial training to learn domain-invariant features:

1. A feature extractor network extracts features from bioimpedance signals
2. A blood pressure estimator network predicts DBP and SBP from these features
3. A domain classifier tries to identify which subject the data came from
4. The feature extractor is trained to maximize blood pressure estimation accuracy while minimizing domain classification accuracy

This adversarial approach forces the model to learn features that are useful for blood pressure estimation but are not specific to individual subjects, enabling better generalization to new subjects with minimal training data.

## Troubleshooting

- If you encounter memory issues, try reducing the batch size by modifying the `batch_size` parameter in the `DataGenerator` class.
- If training is unstable, try adjusting the learning rate or domain adaptation parameter schedule.
- If you receive dimension mismatch errors, ensure that all your data files have consistent shapes and formats.

## Citation

If you use this code in your research, please cite the original paper:

```
Zhang, L., Hurley, N. C., Ibrahim, B., Spatz, E., Krumholz, H. M., Jafari, R., & Mortazavi, B. J. (2020). 
Developing Personalized Models of Blood Pressure Estimation from Wearable Sensors Data Using Minimally-trained 
Domain Adversarial Neural Networks. Proceedings of Machine Learning Research, 126, 97-120.
```
