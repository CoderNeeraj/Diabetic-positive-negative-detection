from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)

# Load the model, scaler, and feature names
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('feature_names.pkl', 'rb') as feature_file:
    feature_names = pickle.load(feature_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the form
    features = [float(x) for x in request.form.values()]
    
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([features], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Transform the numerical features
    numerical_features = scaler.transform(input_data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
    numerical_features_df = pd.DataFrame(numerical_features, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Generate categorical features
    numerical_features_df['BMI_Category_Underweight'] = (numerical_features_df['BMI'] < 18.5).astype(int)
    numerical_features_df['BMI_Category_Normal'] = ((numerical_features_df['BMI'] >= 18.5) & (numerical_features_df['BMI'] < 25)).astype(int)
    numerical_features_df['BMI_Category_Overweight'] = ((numerical_features_df['BMI'] >= 25) & (numerical_features_df['BMI'] < 30)).astype(int)
    numerical_features_df['BMI_Category_Obesity 1'] = ((numerical_features_df['BMI'] >= 30) & (numerical_features_df['BMI'] < 35)).astype(int)
    numerical_features_df['BMI_Category_Obesity 2'] = ((numerical_features_df['BMI'] >= 35) & (numerical_features_df['BMI'] < 40)).astype(int)
    numerical_features_df['BMI_Category_Obesity 3'] = (numerical_features_df['BMI'] >= 40).astype(int)
    numerical_features_df['Insulin_Category_Normal'] = ((numerical_features_df['Insulin'] >= 16) & (numerical_features_df['Insulin'] <= 166)).astype(int)
    numerical_features_df['Glucose_Category_Low'] = (numerical_features_df['Glucose'] <= 70).astype(int)
    numerical_features_df['Glucose_Category_Normal'] = ((numerical_features_df['Glucose'] > 70) & (numerical_features_df['Glucose'] <= 99)).astype(int)
    numerical_features_df['Glucose_Category_Prediabetes'] = ((numerical_features_df['Glucose'] > 99) & (numerical_features_df['Glucose'] <= 126)).astype(int)
    numerical_features_df['Glucose_Category_Diabetes'] = (numerical_features_df['Glucose'] > 126).astype(int)

    # Ensure that the DataFrame columns match the feature names used in the model
    input_features = numerical_features_df.reindex(columns=feature_names, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_features)[0]

    return render_template('index.html', prediction_text='Diabetes Prediction: {}'.format('Positive' if prediction == 1 else 'Negative'))

if __name__ == "__main__":
    app.run(debug=True)
