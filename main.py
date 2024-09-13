import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.cluster import KMeans
from sklearn.calibration import calibration_curve
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Failure Probability Prediction",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the trained model
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=1)

# Load the data
train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(train_data.drop(columns=["Target"]))
X_test_imputed = imputer.transform(test_data.drop(columns=["Target"]))

X_train = pd.DataFrame(X_train_imputed, columns=train_data.drop(columns=["Target"]).columns)
X_test = pd.DataFrame(X_test_imputed, columns=test_data.drop(columns=["Target"]).columns)

y_train = train_data["Target"]
y_test = test_data["Target"]

# Train the model
model.fit(X_train, y_train)

# Function to predict failure probability
def predict_failure_probability(input_values):
    input_df = pd.DataFrame([input_values], columns=X_train.columns)
    return model.predict_proba(input_df)[0][1] * 100  # Multiply by 100 to get percentage

# Function to plot univariate analysis
def plot_univariate_analysis():
    st.subheader("Univariate Analysis")
    for column in X_train.columns[:5]:  # Showing only 5 columns for brevity
        st.write(f"### {column}")
        plt.figure(figsize=(8, 6))
        sns.histplot(train_data[column], kde=True, bins=20, color='skyblue')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        st.pyplot(plt)
        st.write("---")

# Function to plot confusion matrix
def plot_conf_matrix():
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Function to plot additional visualizations
def plot_additional_visualizations():
    st.subheader("Additional Visualizations")

    # ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
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
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    st.pyplot(plt)

# Function to plot feature importance based on model weights
def plot_feature_importance():
    st.subheader("Feature Importance")
    layer_weights = model.coefs_
    feature_importance = np.mean(np.abs(layer_weights[0]), axis=1)  # Using mean absolute weight of input layer

    feature_names = X_train.columns
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance (Weight Magnitude)")
    plt.xlabel("Mean Absolute Weight")
    plt.ylabel("Feature")
    st.pyplot(plt)

# Function to plot learning curve
def plot_learning_curve():
    st.subheader("Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1,
                                                            train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-Validation Score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    st.pyplot(plt)

# Function to display best hyperparameters
def display_best_hyperparameters():
    st.subheader("Best Hyperparameters")
    best_params = {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'}
    st.write(best_params)

# Function to plot calibration curve
def plot_calibration_curve():
    st.subheader("Calibration Curve")
    prob_pos = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='MLP Classifier')
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    st.pyplot(plt)

# Function to plot residual plot
def plot_residual_plot():
    st.subheader("Residual Plot")
    residuals = y_test - model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals)
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Plot")
    st.pyplot(plt)

# Function to perform clustering
def perform_clustering():
    st.subheader("Cluster Analysis")
    kmeans = KMeans(n_clusters=3, random_state=1)
    clusters = kmeans.fit_predict(X_train)

    cluster_df = pd.DataFrame(X_train, columns=X_train.columns)
    cluster_df['Cluster'] = clusters

    st.write(cluster_df.head())

# Function to plot scatter plot
def plot_scatter_plot():
    st.subheader("Scatter Plot")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='viridis')  # Using iloc to select columns
    plt.xlabel(X_train.columns[0])  # Setting x-axis label dynamically
    plt.ylabel(X_train.columns[1])  # Setting y-axis label dynamically
    plt.title("Scatter Plot with Target Labels")
    plt.colorbar(label='Target')
    st.pyplot(plt)

# Function to plot pair plot
def plot_pair_plot():
    st.subheader("Pair Plot")
    # Concatenate X_train and y_train for pair plot
    pair_plot_data = pd.concat([X_train, y_train], axis=1)
    sns.pairplot(pair_plot_data, hue='Target', palette='viridis')
    st.pyplot(plt)

# Main function
def main():
    st.title("Failure Probability Prediction")

    # Load custom CSS styles for better UI
    st.markdown("""
        <style>
        .st-DYHlo.st-joJCcF.st-eVYmi { /* Sidebar container */
            background-color: #f0f0f0;
            border-radius: 10px;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);
        }

        .st-br.st-eiHhjH.st-dQjBJM.st-dhPnNz { /* Main content container */
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);
        }

        .css-1ipymci { /* Section headers */
            color: #333333 !important;
        }

        .css-1u5g2a2 { /* Subheaders */
            color: #444444 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.subheader("Navigation")
    tabs = ["Home", "Prediction", "Current Visualizations"]
    selected_tab = st.sidebar.radio("Select Section", tabs)

    # Main content based on selected tab
    if selected_tab == "Home":
        st.write("Welcome to the Failure Probability Prediction App!")
        st.write("This app helps in predicting the probability of failure based on input features.")
        st.write("Use the sidebar to navigate through different sections.")

    elif selected_tab == "Prediction":
        st.write("## Prediction Section")
        st.write("Adjust sliders for V1 to V40 to predict the probability of failure.")

        input_values = []
        for i in range(1, 41):
            input_values.append(st.sidebar.slider(f"V{i}", min_value=-10.0, max_value=10.0, value=0.0, step=0.01))

        if st.sidebar.button("Predict"):
            failure_probability = predict_failure_probability(input_values)
            st.write(f"Probability of Failure: {failure_probability:.2f}%")  # Display as percentage

        # Plot line chart
        st.write("## Prediction Line Chart")
        st.write("This chart shows the predicted probability of failure over time.")
        time_values = np.linspace(0, 10, 100)  # Example time values
        prob_values = np.random.rand(100) * 100  # Example probability values
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, prob_values)
        plt.xlabel("Time")
        plt.ylabel("Probability (%)")
        plt.title("Prediction Line Chart")
        st.pyplot(plt)

    elif selected_tab == "Current Visualizations":
        st.write("## Current Visualizations")
        st.write("Choose from the following visualizations:")

        visualization_option = st.selectbox("Select Visualization",
                                            ("Univariate Analysis", "Confusion Matrix", "Additional Visualizations",
                                             "Feature Importance", "Learning Curve", "Best Hyperparameters",
                                             "Calibration Curve", "Residual Plot", "Cluster Analysis",
                                             "Scatter Plot", "Pair Plot"))  # Added "Pair Plot"

        # Call corresponding function based on selected visualization
        if visualization_option == "Univariate Analysis":
            plot_univariate_analysis()
        elif visualization_option == "Confusion Matrix":
            plot_conf_matrix()
        elif visualization_option == "Additional Visualizations":
            plot_additional_visualizations()
        elif visualization_option == "Feature Importance":
            plot_feature_importance()
        elif visualization_option == "Learning Curve":
            plot_learning_curve()
        elif visualization_option == "Best Hyperparameters":
            display_best_hyperparameters()
        elif visualization_option == "Calibration Curve":
            plot_calibration_curve()
        elif visualization_option == "Residual Plot":
            plot_residual_plot()
        elif visualization_option == "Cluster Analysis":
            perform_clustering()
        elif visualization_option == "Scatter Plot":
            plot_scatter_plot()
         # Call the pair plot function

# Execute the main function
if __name__ == "__main__":
    main()