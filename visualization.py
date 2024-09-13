import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def plot_univariate_analysis(train_data):
    st.subheader("Univariate Analysis")
    for column in train_data.columns:
        st.write(f"### {column}")
        plt.figure(figsize=(8, 6))
        sns.histplot(train_data[column], kde=True, bins=20, color='skyblue')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        st.pyplot(plt)
        st.write("---")

def plot_conf_matrix(y_test, predicted_labels):
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

def plot_additional_visualizations(y_test, predicted_probabilities):
    st.subheader("Additional Visualizations")

    # ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, predicted_probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # Precision-Recall Curve
    st.write("### Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, predicted_probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    st.pyplot(plt)