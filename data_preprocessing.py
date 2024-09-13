import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(train_data, test_data):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(train_data.drop(columns=["Target"]))
    X_test_imputed = imputer.transform(test_data.drop(columns=["Target"]))

    X_train = pd.DataFrame(X_train_imputed, columns=train_data.drop(columns=["Target"]).columns)
    X_test = pd.DataFrame(X_test_imputed, columns=test_data.drop(columns=["Target"]).columns)

    y_train = train_data["Target"]
    y_test = test_data["Target"]
    
    return X_train, X_test, y_train, y_test