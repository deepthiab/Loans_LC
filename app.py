# imports
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify
import os
import xgboost as xgb
import logging

# Configure the logger
logging.basicConfig(
    filename='flask_app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask('myApp')

@app.route('/')
def home():
    logger.info('Input form rendered')
    return render_template('form.html')

@app.route("/submit")
def make_predictions():
    logger.info('Data route accessed with method %s', request.method)
    # load the form data from the incoming request
    user_input = request.args
    c_policy = np.array(bool(user_input['credit.policy']))
    int_rate = np.array(float(user_input['int.rate']))
    installment = np.array(float(user_input['installment']))
    annual_inc = (int(user_input['annual.inc'])*1000)
    log_annual_inc = np.array(np.log(annual_inc))
    dti = np.array(float(user_input['dti']))
    fico = np.array(int(user_input['fico']))
    cr_line = np.array(float(user_input['days.with.cr.line']))
    r_bal = np.array(int(user_input['revol.bal']))
    r_util = np.array(float(user_input['revol.bal']))
    inq = np.array(int(user_input['inq.last.6mths']))
    delinq = np.array(int(user_input['delinq.2yrs']))
    pub_rec = np.array(int(user_input['pub.rec']))
    x0_all_other , x0_credit_card , x0_debt_consolidation , x0_educational , x0_home_improvement , x0_major_purchase , x0_small_business = 0,0,0,0,0,0,0
    if user_input['purpose']== 'credit_card':
        x0_credit_card = 1
    if user_input['purpose']== 'debt_consolidation':
        x0_debt_consolidation = 1
    if user_input['purpose']==  'educational':
        x0_educational =1
    if user_input['purpose']==  'home_improvement':
        x0_home_improvement = 1
    if user_input['purpose']== 'major_purchase':
        x0_major_purchase = 1
    if user_input['purpose']==  'small_business':
        x0_small_business = 1
    else:
            x0_all_other = 1
    purpose = np.array( [x0_all_other , x0_credit_card, x0_debt_consolidation, x0_educational,x0_home_improvement ,x0_major_purchase , x0_small_business], dtype=float
                       )
    data = np.hstack((c_policy, int_rate , installment , log_annual_inc , dti , fico , cr_line , r_bal , r_util , inq , delinq , pub_rec , purpose),dtype=object)
    
    data = data.reshape(1,-1)
    print(data.shape)
    cwd = os.getcwd()
    file_path = os.path.join(cwd,'xgb_gs_model.p')
    xg_filepath = os.path.join(cwd,'best_xgbgs.json')
    ss_filepath = os.path.join(cwd,'std_scaler.pkl')
    with open('std_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    scaled_input = scaler.transform(data)        
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(xg_filepath)
    prediction = loaded_model.predict(scaled_input)
    if prediction:
        result = 'likely'
    else:
        result = 'unlikely'

    #return render_template("results.html", uri = round(prediction, 2))
    return render_template("results.html", result = result)

if __name__ == "__main__":
    logger.info('Starting Flask app')
    app.run(debug=True)
    