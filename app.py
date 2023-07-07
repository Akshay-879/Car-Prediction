import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.decomposition import PCA

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('trail.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    
    # Load the pre-fitted PCA model
    pca = pickle.load(open('pca_model.pkl', 'rb'))

    # Transform the single data point using the pre-fitted PCA model
    X = pca.transform([int_features])

    # Make prediction
    prediction = model.predict(X)

#     output = round(prediction[0], 2)

    return render_template('trail.html', prediction_text='Its a '.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)