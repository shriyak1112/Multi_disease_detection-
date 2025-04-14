from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import time
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
brain_model = load_model('models/brain_tumor.h5')
skin_model = load_model('models/skin_cancer_final.h5')
parkinsons_model = joblib.load('models/parkinsons_xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')  # Load the scaler for Parkinson's data

# Load heart disease model and scaler
heart_model = joblib.load('models/heart_model.pkl')
scaler = joblib.load('models/scaler_H.pkl')

diabetes_model = joblib.load('models/diabetes_model.pkl')
scaler = joblib.load('models/scaler_D.pkl')

# Image preprocessing function
def preprocess_image(image_path, target_size=(299, 299)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Prediction function for image-based diseases
def predict_disease(model, image_path, disease):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    
    if disease == 'brain_tumor':
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        predicted_class = np.argmax(prediction)
        return f"{classes[predicted_class]}"
    elif disease == 'skin_cancer':
        return "Malignant" if prediction[0][0] > 0.5 else "Benign"
    else:
        return "Unknown disease type"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect/<disease>', methods=['GET', 'POST'])
def detect(disease):
    result = None
    image_path = None
    relative_image_path = None
    
    if disease == 'parkinsons':
        return redirect(url_for('parkinsons'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('detect.html', disease=disease, error="No file part")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('detect.html', disease=disease, error="No selected file")
        
        if file:
            filename = str(int(time.time())) + '_' + file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            relative_image_path = 'uploads/' + filename
            
            if disease == 'brain_tumor':
                model = brain_model
            elif disease == 'skin_cancer':
                model = skin_model
            else:
                return render_template('detect.html', disease=disease, error="Invalid disease type")
            
            result = predict_disease(model, image_path, disease)
    
    display_name = ' '.join(word.capitalize() for word in disease.split('_'))
    
    return render_template('detect.html', 
                           disease=disease,
                           display_name=display_name,
                           result=result, 
                           image_path=relative_image_path)

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    result = None
    input_values = {}
    
    if request.method == 'POST':
        features = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
        
        # Collect user input
        input_values = {feature: float(request.form.get(feature, 0)) for feature in features}
        input_data = np.array([list(input_values.values())]).astype(float)

        # Scale and predict
        input_data_scaled = scaler.transform(input_data)
        prediction = parkinsons_model.predict(input_data_scaled)

        # Interpret result
        if prediction[0] == 1:
            result = "The person has Parkinson's Disease"
        else:
            result = "The person does NOT have Parkinson's Disease"
    
    return render_template('parkinsons.html', result=result, input_values=input_values)

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    input_values = {}
    
    if request.method == 'POST':
        features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
            'AgeGroup', 'BMI_Category', 'Glucose_Insulin_Ratio',
            'Age_BMI_Product', 'Glucose_Squared'
        ]

        input_values = {feature: float(request.form.get(feature, 0)) for feature in features}
        input_array = np.array([list(input_values.values())])
        input_scaled = scaler.transform(input_array)

        prediction = diabetes_model.predict(input_scaled)[0]
        
        result = "🩺 The person is likely **Diabetic**" if prediction == 1 else "✅ The person is likely **Not Diabetic**"
    
    return render_template('diabetes.html', result=result, input_values=input_values)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    result = None
    input_values = {}

    if request.method == 'POST':
        features = [
            'age', 'trestbps', 'chol', 'thalch', 'oldpeak',
            'sex_Male', 'sex_Female',
            'cp_typical angina', 'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal pain',
            'fbs_True', 'fbs_False',
            'restecg_left ventricular hypertrophy', 'restecg_normal', 'restecg_ST-T wave abnormality',
            'exang_True', 'exang_False',
            'slope_downsloping', 'slope_flat', 'slope_upsloping',
            'ca_0.0', 'ca_1.0', 'ca_2.0', 'ca_3.0', 'ca_4.0',
            'thal_fixed defect', 'thal_normal', 'thal_reversible defect'
        ]

        # Get numeric values
        numeric_fields = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        encoded_fields = [f for f in features if f not in numeric_fields]

        # Gather input
        for field in numeric_fields:
            input_values[field] = float(request.form.get(field, 0))

        for field in encoded_fields:
            input_values[field] = 1.0 if request.form.get(field) == 'on' else 0.0

        # Convert to array
        input_array = np.array([list(input_values.values())])
        input_scaled = heart_scaler.transform(input_array)

        prediction = heart_model.predict(input_scaled)[0]
        result = "💔 Likely to have Heart Disease" if prediction == 1 else "❤️ Unlikely to have Heart Disease"

    return render_template('heart.html', result=result, input_values=input_values)

if __name__ == '__main__':
    app.run(debug=True)
