%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv

import pandas as pd

df = pd.read_csv('parkinsons.csv')
print(df.columns)

selected_features = ['DFA', 'PPE']
output_feature = 'status'

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[selected_features] = scaler.fit_transform(df[selected_features])

from sklearn.model_selection import train_test_split

X = df[selected_features]
y = df[output_feature]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Initialize the SVM model with an RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib

joblib.dump(svm_model, 'my_model.joblib')
