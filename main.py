from flask import Flask, render_template, request
from joblib import load

import numpy as np

from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("About.html")


@app.route("/kidney")
def kidney():
    return render_template("Kidney.html")


@app.route("/kidney_predict", methods=["POST"])
def kidney_predict():
    # Loading the model of XGBoast
    model = load('Models_joblib/kidney_Model.joblib')

    # Taking the Inputs from the Form using POST Method and converting to float
    if request.method == 'POST':
        white_blood_cell_count = float(request.form.get('white_blood_cell_count'))
        blood_glucose_random = float(request.form.get('blood_glucose_random'))
        blood_urea = float(request.form.get('blood_urea'))
        serum_creatinine = float(request.form.get('serum_creatinine'))
        packed_cell_volume = float(request.form.get('packed_cell_volume'))
        albumin = float(request.form.get('albumin'))
        haemoglobin = float(request.form.get('haemoglobin'))
        age = float(request.form.get('age'))
        sugar = float(request.form.get('sugar'))
        hypertension = float(request.form.get('hypertension'))

        # Printing the inputs to check inputs
        # print(white_blood_cell_count, blood_glucose_random, blood_urea, serum_creatinine, packed_cell_volume, albumin,
        #       haemoglobin, age, sugar, hypertension)

        # Prediction features
        features = np.array(
            [[white_blood_cell_count, blood_glucose_random, blood_urea, serum_creatinine, packed_cell_volume, albumin,
              haemoglobin, age, sugar, hypertension]])

        # Predicting the Resulting using model and converting to int
        results_predict = int(model.predict(features))

        # Printing the results for understanding
        # print(results_predict)

        # According to results redirecting to results page
        if results_predict:
            prediction = "There are chances of Chronic Kidney Disease ! Consult your doctor Soon."

        else:
            prediction = "No need to fear. You have no dangerous symptoms of the Chronic Kidney Disease. For Better " \
                         "Understanding you can consult your doctor! "

    return render_template("result.html", prediction_text=prediction)


@app.route("/diabetes")
def diabetes():
    return render_template("Diabetes.html")


@app.route("/diabetes_predict", methods=["POST"])
def diabetes_predict():
    model = load('Models_joblib/diabetes_Model.joblib')

    # Taking the Inputs from the Form using POST Method and converting to float
    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        # print(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)

        # Prediction features
        features = np.array(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Predicting the Resulting using model and converting to int
        results_predict = int(model.predict(features))

        # Printing the results for understanding
        print(results_predict)

        # According to results redirecting to results page
        if results_predict:
            prediction = "There are chances of Diabetes ! Consult your doctor Soon."
            # print("There are chances of Diabetes ! Consult your doctor Soon.")

        else:
            prediction = "No need to fear. You have no dangerous symptoms of the Diabetes. For Better Understanding " \
                         "you can consult your doctor! "
            # print("No diabetes Chances")

    return render_template("result.html", prediction_text=prediction)


@app.route("/heart")
def heart():
    # return "You are on Heart Disease Detection Page"
    return render_template("heart.html")


@app.route("/heart_predict", methods=["POST"])
def heart_predict():
    # Loading the model of Random Forest Classifier Model
    # model = load('Models_joblib/Heart_model.joblib')
    model = load('Models_joblib/Heart_model_15_features.joblib')

    # Taking the Inputs from the Form using POST Method and converting to float
    if request.method == 'POST':
        age = float(request.form.get('age'))
        resting_blood_pressure = float(request.form.get('resting_blood_pressure'))
        cholesterol = float(request.form.get('cholesterol'))
        fasting_blood_sugar = float(request.form.get('fasting_blood_sugar'))
        max_heart_rate_achieved = float(request.form.get('max_heart_rate_achieved'))
        exercise_induced_angina = float(request.form.get('exercise_induced_angina'))
        st_depression = float(request.form.get('st_depression'))

        gender = int(request.form.get('gender'))
        if gender:
            sex_male = 1
        else:
            sex_male = 0

        chest_pain_type = int(request.form.get('chest_pain_type'))
        if chest_pain_type == 1:
            chest_pain_type_Typical_angina = 1
            chest_pain_type_Atypical_angina = 0
            chest_pain_type_Non_angina = 0
        elif chest_pain_type == 2:
            chest_pain_type_Typical_angina = 0
            chest_pain_type_Atypical_angina = 1
            chest_pain_type_Non_angina = 0
        elif chest_pain_type == 1:
            chest_pain_type_Typical_angina = 0
            chest_pain_type_Atypical_angina = 0
            chest_pain_type_Non_angina = 1
        else:
            chest_pain_type_Typical_angina = 0
            chest_pain_type_Atypical_angina = 0
            chest_pain_type_Non_angina = 0

        rest_ecg = int(request.form.get('rest_ecg'))
        if rest_ecg == 0:
            rest_ecg_left_ventricular_hypertrophy = 0
            rest_ecg_normal = 1
        elif rest_ecg == 2:
            rest_ecg_left_ventricular_hypertrophy = 1
            rest_ecg_normal = 0
        else:
            rest_ecg_left_ventricular_hypertrophy = 0
            rest_ecg_normal = 0

        st_slope = int(request.form.get('st_slope'))
        if st_slope == 1:
            st_slope_flat = 0
            st_slope_upsloping = 1
        elif st_slope == 2:
            st_slope_flat = 1
            st_slope_upsloping = 0
        else:
            st_slope_flat = 0
            st_slope_upsloping = 0

        # Printing the inputs to check inputs
        print(age, resting_blood_pressure, cholesterol, fasting_blood_sugar, max_heart_rate_achieved,
              exercise_induced_angina, st_depression, sex_male, chest_pain_type_Atypical_angina,
              chest_pain_type_Non_angina, chest_pain_type_Typical_angina, rest_ecg_left_ventricular_hypertrophy,
              rest_ecg_normal, st_slope_flat, st_slope_upsloping)

        # Prediction features
        features = np.array(
            [[age, resting_blood_pressure, cholesterol, fasting_blood_sugar, max_heart_rate_achieved,
              exercise_induced_angina, st_depression, sex_male, chest_pain_type_Atypical_angina,
              chest_pain_type_Non_angina, chest_pain_type_Typical_angina, rest_ecg_left_ventricular_hypertrophy,
              rest_ecg_normal, st_slope_flat, st_slope_upsloping]])

        # Predicting the Results using model and converting to int
        results_predict = int(model.predict(features))

        # Printing the results for understanding
        print(results_predict)

        # According to results redirecting to results page
        if results_predict:
            prediction = "There are chances of Heart Disease ! Consult your doctor Soon."

        else:
            prediction = "No need to fear. You have no dangerous symptoms of the Heart Disease. For Better " \
                         "Understanding you can consult your doctor! "

    return render_template("result.html", prediction_text=prediction)


app.run(debug=True)
