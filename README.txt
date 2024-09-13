# Failure Probability Prediction App

## Overview

This project involves building a machine learning model to predict failure probabilities based on input features. The app is built using Streamlit and provides various visualizations to understand model performance and data distributions. 

---

## Project Structure

```
.
├── main.py                # Streamlit app for failure probability prediction and visualizations
├── data_preprocessing.py   # Script for cleaning and preparing the data
├── model_training.py       # Script for training the machine learning model
├── visualization.py        # Script for advanced visualizations
├── Train.csv               # Training dataset
├── Test.csv                # Testing dataset
└── README.md               # Project documentation
```

---

## Requirements

Install the required dependencies using the following command:

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn numpy
```

---

## Running the App

To run the app, follow these steps:

1. Clone the repository.
2. Install the required libraries.
3. Run the Streamlit app using the following command:

```bash
streamlit run main.py
```

This will open the app in your web browser.

---

## Features

1. **Prediction**:
   - Input feature values using sliders to predict the probability of failure.
   - Displays the failure probability as a percentage.

2. **Visualizations**:
   - **Univariate Analysis**: View distributions of selected features.
   - **Confusion Matrix**: Evaluate model performance.
   - **ROC Curve and Precision-Recall Curve**: Assess classification performance.
   - **Feature Importance**: Visualize the significance of each feature.
   - **Learning Curve**: Understand how the model performs with different training data sizes.
   - **Cluster Analysis**: Perform KMeans clustering on the dataset.
   - **Scatter and Pair Plots**: Visualize feature relationships.

---

## Dataset

- `Train.csv`: Contains the training data used for model development.
- `Test.csv`: Contains the testing data for evaluating the model.

---
