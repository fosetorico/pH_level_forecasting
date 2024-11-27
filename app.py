from flask import Flask,request,render_template, jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
@cross_origin()
def index(): 
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            Temp=float(request.form.get('Temp')),
            SEC=float(request.form.get('SEC')),
            Turbidity=float(request.form.get('Turbidity')),
            Total_Iron=float(request.form.get('Total_Iron')),
            Titration_1=float(request.form.get('Titration_1')),
            Titration_2=float(request.form.get('Titration_2')),
            Volume=request.form.get('Volume'),
            N_VALUE=float(request.form.get('N_VALUE')),
            Tryptophan_Probe=float(request.form.get('Tryptophan_Probe')),
            Final_HCO3=float(request.form.get('Final_HCO3')),
        ) 
        pred_df=data.get_data_as_data_frame()
        print(pred_df)        

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('index.html',results=round(results[0], 2))
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True) 
    # app.run(host="0.0.0.0") 