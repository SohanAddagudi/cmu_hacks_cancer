from django.shortcuts import render
from django import forms
import pandas as pd
import joblib
import os
from django.conf import settings
from sklearn.preprocessing import StandardScaler

# Define the form
class DataUploadForm(forms.Form):
    file = forms.FileField(label='Select your CSV file')

# Load the saved model and scaler from disk (once when the app starts)
model_path = os.path.join(settings.BASE_DIR, '/Users/sa/Dev/cancer/knn_model.pkl')
scaler_path = os.path.join(settings.BASE_DIR, '/Users/sa/Dev/cancer/scaler.pkl')

try:
    knn = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    knn = None
    scaler = None
    print(f"Error loading model or scaler: {e}")

def predict(request):
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Read the uploaded CSV file
                uploaded_file = request.FILES['file']
                data = pd.read_csv(uploaded_file)

                # Preprocess the uploaded data
                data = data.rename(columns={'Unnamed: 0': 'patient_id'})
                data = data.set_index('patient_id')
                data = data.reset_index(drop=True)
                data = data.drop(['patient_id'], axis=1, errors='ignore')

                # Standardize the input data using the saved scaler
                data_scaled = scaler.transform(data)

                # Predict using the loaded model
                predictions = knn.predict(data_scaled)

                cancer_map = {
                    'Adrenocortical Carcinoma': 0, 'Bladder Urothelial Carcinoma': 1, 'Breast Invasive Carcinoma': 2, 'Cervical Squamous Cell Carcinoma': 3, 'Cholangiocarcinoma': 4, 'Colon Adenocarcinoma': 5, 'Diffuse Large B-cell Lymphoma': 6, 'Esophageal Carcinoma': 7, 'Glioblastoma Multiforme': 8,
                    'Head and Neck Squamous Cell Carcinoma': 9, 'Kidney Chromophobe': 10, 'Kidney Renal Cell Carcinoma': 11, 'Kidney Renal Papillary Cell Carcinoma': 12, 'Acute Myeloid Leukemia': 13, 'Brain Lower Grade Glioma': 14, 'Liver Hepatocellular Carcinoma': 15, 'Lung Adenocarcinoma': 16, 
                    'Lung Squamous Cell Carcinoma': 17, 'Mesothelioma': 18, 'Ovarian Serous Cystadenocarcinoma': 19, 'Pancreatic Adenocarcinoma': 20, 'Pheochromocytoma and Paraganglioma': 21, 'Prostate Adenocarcinoma': 22, 'Rectum Adenocarcinoma': 23, 'Sarcoma': 24, 
                    'Skin Cutaneous Melanoma': 25, 'Stomach Adenocarcinoma': 26, 'Testicular Germ Cell Tumors': 27, 'Thyroid Carcinoma': 28, 'Thymoma': 29, 'Uterine Corpus Endometrial Carcinoma': 30, 'Uterine Carcinosarcoma': 31, 'Uveal Melanoma': 32, 
                    'Normal   ': 33
                }
                reverse_cancer_map = {v: k for k, v in cancer_map.items()}
                cancers = []
                for i in predictions:
                    for x in range(len(i)):
                        if i[x] == 1:
                            cancers.append(reverse_cancer_map[x])
                length = len(predictions)
                # print(predictions.shape)
                zipped_data = zip(predictions, cancers)
                if predictions.shape[0] > 1:
                # Return the predictions to the template
                    return render(request, 'predictor/result.html', {'predictions': predictions, 'cancers': cancers, 'zipped_data': zipped_data})
                else:
                    return render(request, 'predictor/oneres.html', {'predictions': predictions, 'cancer': cancers[0], 'zipped_data': zipped_data})
            except Exception as e:
                # Handle the error by rendering error.html and passing the error message
                return render(request, 'predictor/error.html', {'error_message': str(e)})
    else:
        form = DataUploadForm()

    return render(request, 'predictor/upload.html', {'form': form})

def heatmaps_page(request):
    return render(request, 'predictor/heatmaps.html')

def cancinfo(request):
    return render(request, 'predictor/cancinfo.html')