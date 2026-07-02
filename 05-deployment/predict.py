import pickle

from flask import Flask
from flask import request
from flask import jsonify

input_file = 'model_C=1.0.bin'

with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

app = Flask('churn-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)


# # Making requests

# # In[ ]:


# import requests


# # In[ ]:


# url = 'http://localhost:9696/predict'


# # In[ ]:


# customer = {
#     'gender': 'female',
#     'seniorcitizen': 0,
#     'partner': 'yes',
#     'dependents': 'no',
#     'phoneservice': 'no',
#     'multiplelines': 'no_phone_service',
#     'internetservice': 'dsl',
#     'onlinesecurity': 'no',
#     'onlinebackup': 'yes',
#     'deviceprotection': 'no',
#     'techsupport': 'no',
#     'streamingtv': 'no',
#     'streamingmovies': 'no',
#     'contract': 'two_year',
#     'paperlessbilling': 'yes',
#     'paymentmethod': 'electronic_check',
#     'tenure': 1,
#     'monthlycharges': 29.85,
#     'totalcharges': 29.85
# }


# # In[ ]:


# response = requests.post(url, json=customer).json()


# # In[ ]:


# response


# # In[ ]:


# if response['churn']:
#     print('sending email to', 'asdx-123d')