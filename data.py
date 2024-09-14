import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the X and y CSV files
X = pd.read_csv('/Users/sa/Dev/cancer/x_data.csv')
y = pd.read_csv('/Users/sa/Dev/cancer/y_data.csv')

#data preprocessing
X = X.rename(columns={'Unnamed: 0': 'patient_id'})
X = X.set_index('patient_id')
y = y.rename(columns={'Unnamed: 0': 'patient_id'})
y = y.set_index('patient_id')

# Reset the indexes so col,rows are aligned
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
X = X.drop(['patient_id'], axis=1, errors='ignore')
y = y.squeeze()

# #split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

# #standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# for k in ([1]):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train_scaled, y_train)
#     y_pred = knn.predict(X_test_scaled)
#     print(f'y_test shape: {y_test.shape}, y_pred shape: {y_pred.shape}')

#     print(f'K={k} - Accuracy: {accuracy_score(y_test, y_pred):.4f}')
#     print(classification_report(y_test, y_pred))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_scaled, y_train)

# Save the trained model and scaler using joblib
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")





