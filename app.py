from flask import Flask, render_template,request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('NBayes.pkl', 'rb'))

@app.route('/home')
def main():
    return render_template('Main.html')


@app.route('/predict',methods=['POST'])
def predict():
    higienis=request.form['higienis']
    pelayanan=request.form['pelayanan']
    variasi=request.form['variasi']
    input={
        "Menurut anda koperasi FST yang baru terlihat lebih higienis dibanding yang lama":higienis,
        "Menurut anda koperasi FST yang baru lebih baik pelayanannya dibanding yang lama":pelayanan,
        "Menurut anda ketersediaan dagangan koperasi FST yang baru lebih bervariasi dibandingkan yang sebelumnya":variasi
    }
    pred = pd.DataFrame([input])
    pred=model.predict(pred)
    return render_template('Main.html', data=pred)
# @app.route('/hello')
# def hello():
#     return 'Hello, World'