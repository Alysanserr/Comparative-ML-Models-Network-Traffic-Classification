# Network Traffic Classification Using Machine Learning and Deep Learning

This repository provides a comprehensive framework for network traffic classification using the CIC-IDS2017 dataset. It implements both classical machine learning and deep learning models to classify network flows into benign or malicious categories. The project includes data preprocessing, feature selection, dimensionality reduction, model training, evaluation, and deployment via a GUI.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Cross-Validation](#cross-validation)
- [Prediction GUI](#prediction-gui)
- [Model Export](#model-export)
- [Results and Insights](#results-and-insights)
- [Usage](#usage)
- [License](#license)

## Project Overview

Network traffic classification is a critical component of cybersecurity, intrusion detection systems (IDS), and network performance monitoring. By accurately classifying network flows, organizations can detect anomalies, prevent attacks, and optimize network performance.

This project leverages the CIC-IDS2017 dataset, a realistic network traffic dataset that captures multiple types of attacks in a controlled environment.

The workflow includes:

- Loading and preprocessing the dataset, including cleaning, normalization, feature selection, and dimensionality reduction.
- Training multiple models, including Logistic Regression as a baseline and various Deep Neural Network (DNN) architectures.
- Evaluating models using metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Selecting the best-performing model based on multiple criteria.
- Deploying a Tkinter GUI for interactive predictions.

## Features

This project provides the following features:

- Comprehensive data preprocessing pipeline including scaling and PCA.
- Implementation of Logistic Regression and multiple DNN architectures, allowing comparison between classical ML and deep learning methods.
- Detailed evaluation metrics, including per-class precision, recall, and F1-score.
- Visualizations of model performance such as confusion matrices, ROC curves, learning curves, and comparative bar plots.
- Cross-validation to ensure robustness and generalization of models.
- Interactive GUI for predictions via form inputs or .txt files.
- Serialization of models and preprocessing objects for deployment without retraining.

## Installation

To install and run this project:

1. Clone the repository:

```bash
git clone https://github.com/Alysanserr/Comparative-ML-Models-Network-Traffic-Classification.git
cd Comparative-ML-Models-Network-Traffic-Classification
````

2. Create a Python virtual environment and activate it:
   
```bash
conda create -n ntc python=3.11
conda activate ntc
````

3. Install dependencies:
Ensure the following packages are installed:

    - numpy – for numerical operations
    
    - pandas – for data manipulation
    
    - scikit-learn – for classical machine learning, scaling, PCA, and metrics
    
    - tensorflow / keras – for deep learning models
    
    - matplotlib & seaborn – for visualization
    
    - tkinter – for the GUI

# Dataset

This project uses the CIC-IDS2017 dataset, one of the most widely recognized datasets for intrusion detection research.

## Original Dataset

- Contains 2,830,743 network flows with 78 features.
- Features include:
  - IP addresses of source and destination
  - Ports and protocols
  - Timestamps
  - Packet lengths and inter-arrival times
  - TCP flags and flow statistics
  - Attack labels (multi-class)

## Preprocessed Version

To optimize for machine learning, the project uses "CICIDS2017: Cleaned & Preprocessed" (available on Kaggle, by Eric Anacleto Ribeiro).

**Key preprocessing steps:**

- **CSV Consolidation** – Merges multiple CSVs into a single dataset for easier processing.
- **Duplicate Removal** – Eliminates both duplicate rows and columns to prevent bias.
- **Handling Missing and Infinite Values** – Infinite values replaced with NaN; rows with missing values (<1%) removed.
- **Column Name Normalization** – Removes spaces, standardizes column names, ensures consistent parsing.
- **Feature Selection** –
  - Remove columns with a single value.
  - Remove highly correlated features (ρ ≥ 0.99).
  - Use H-statistics and Random Forest importance to retain only informative features.
- **Target Transformation** –
  - Rename Label → Attack Type.
  - Group similar attacks (e.g., DoS Hulk & DoS GoldenEye → DoS).
  - Remove rare attacks like Heartbleed and Infiltration.

After preprocessing, the dataset contains 52 features representing critical network traffic behavior and labels for classification.

# Data Preprocessing

Preprocessing ensures high-quality inputs for models. Steps include:

- **Scaling**: StandardScaler normalizes all features to zero mean and unit variance, ensuring that no single feature dominates learning.
- **Dimensionality Reduction (PCA)**:
  - Removes correlated or redundant features.
  - Reduces feature space while preserving variance.
  - Improves training speed and model generalization.
- **Target Encoding**:
  - DNNs: one-hot encoding of categorical labels.
  - Logistic Regression: integer labels.
- **Train-Test Split**:
  - PCA and scaler are fitted only on the training set to prevent data leakage.
  - The test set provides unbiased evaluation.
- **Feature Justification**:
  - Selected 52 features encapsulate essential traffic characteristics such as:
    - Packet length statistics
    - Flow durations and rates
    - TCP flag counts
    - Inter-arrival time metrics

# Model Training

## Logistic Regression

- Serves as a baseline.
- Hyperparameter tuning via grid search.
- Advantages: interpretable, fast, suitable for linearly separable patterns.

## Deep Neural Networks (DNNs)

Three architectures evaluated:

### RN1 (Best Performing)
- 2 hidden layers: 20 → 10 neurons
- ReLU activation in hidden layers
- Softmax output
- 50 epochs, batch size 500, Adam optimizer

### RN2
- 4 hidden layers: 20 → 30 → 20 → 10 neurons
- 30 epochs, batch size 1000

### RN3
- Single hidden layer: 20 neurons
- 30 epochs, batch size 1000

**Common parameters:**
- Loss function: categorical_crossentropy
- Metric: accuracy

# Model Evaluation

Models evaluated using:

- **Accuracy**: Overall correct predictions
- **Precision**: Proportion of true positives among predicted positives
- **Recall**: Proportion of true positives detected among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Confusion Matrix**: Detailed per-class prediction performance

**Visualizations:**
- Learning curves (accuracy & loss per epoch)
- Confusion matrices per model
- Comparative bar plots for precision, recall, F1-score
- Boxplots to visualize model stability across multiple runs

# Cross-Validation

- K-Fold Cross-Validation (k=5) applied to RN1.
- Ensures model generalization and robustness.
- Accuracy per fold: [0.9567, 0.9589, 0.9054, 0.9685, 0.9182]
- Mean accuracy: 0.9416, indicating strong stability.

# Prediction GUI

Tkinter-based GUI allows:

- Manual input of 52 feature values
- Uploading .txt files with comma-separated values

**Features:**
- Scrollable interface for all features
- Error handling for missing or invalid data
- Real-time model prediction display
- Option to choose between models (Logistic Regression or DNN)

# Model Export

Models and preprocessing objects serialized via pickle for inference:

- Trained DNN (RN1)
- Logistic Regression
- Scaler object
- PCA object
- Class labels

This allows deployment without retraining.

# Results and Insights

- **RN1**: Best overall performance; high F1-score and consistent accuracy
- **RN2**: Comparable but slower; better for complex patterns
- **RN3**: Fast but slightly lower accuracy
- **Logistic Regression**: Baseline; interpretable but less effective on rare attack classes
- **Visualization**: Confirms RN1 superiority in detecting DoS and Brute Force attacks
- **PCA** improved convergence and reduced training time without significant loss in accuracy

# Usage

This project provides a Graphical User Interface (GUI) to predict network traffic activity types using pre-trained machine learning models (Logistic Regression and Neural Network). You can provide inputs manually through the form or load a `.txt` file containing the 52 features.

## 1. Launching the GUI

To start the GUI, simply run:

```bash
python gui_predictor.py
````

The main window will appear with:

- **Scrollable form:** 52 input fields corresponding to traffic features.

- **Buttons:**
  - **"Predict from Form":** Predicts the traffic type using the values entered manually.
  - **"Load .txt file with 52 features":** Allows uploading a `.txt` file containing all feature values separated by commas.

  ![Image](https://github.com/user-attachments/assets/156a41b2-53ff-4450-81f6-7b699d8b84bf)
  
## 2. Providing Input via Form

Enter numeric values for all 52 traffic features, such as:

```bash
Destination Port, Flow Duration, Total Fwd Packets, ..., Idle Min
````

Click **"Predict from Form"**.

The system will validate the input:

- Must contain exactly 52 numeric values.
- Any non-numeric or missing input will trigger an error popup.
  
## 3. Providing Input via .txt File

Prepare a `.txt` file with 52 comma-separated values, matching the feature order. Example:

```bash
443, 1200, 20, 15000, ..., 0.05
````
Click "Load .txt file with 52 features" in the GUI.

The program validates the file:
    - Ensures exactly 52 values.
    - Displays an error message if validation fails.
    - Otherwise, performs the prediction automatically.

## 4. How Predictions Work

Once input is submitted (manually or via file), the GUI internally:

- Reshapes the input into a 1×52 vector.
- Scales features using the pre-trained scaler.
- Applies PCA transformation and selects the first 10 components, matching the neural network input.
- Each model predicts probabilities for all traffic categories:
  - **Logistic Regression:** outputs via `predict_proba`.
  - **Neural Network:** outputs directly.
- The predicted class is selected as the category with highest probability.
- A graph is generated for each model showing the probability distribution across all categories, with:
  - **X-axis:** Traffic categories
  - **Y-axis:** Probability
  - Blue markers indicating each class probability

  ![Image](https://github.com/user-attachments/assets/34209b34-4a82-442f-956e-ba348c20c921)

## 5. Notes

- Models are pre-trained and serialized, so no training is required to make predictions.
- Ensure your inputs are preprocessed similarly to the training data (52 numeric features in the correct order).
- The GUI automatically applies scaling and PCA, so raw feature values can be used directly.
- This setup allows users to quickly test network traffic flows without running heavy preprocessing pipelines.

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

