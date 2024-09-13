import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

def train_model(train_data):
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=1)
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(train_data.drop(columns=["Target"]))
    X_train = pd.DataFrame(X_train_imputed, columns=train_data.drop(columns=["Target"]).columns)
    
    y_train = train_data["Target"]
    
    model.fit(X_train, y_train)
    
    return model