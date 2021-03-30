import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("smp.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    featurs_name = ['study_hours']
    df = pd.DataFrame(features_value, columns=featurs_name)
    output = model.predict(df)

    return render_template('index.html', prediction_text='Student will obtain {} marks'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
