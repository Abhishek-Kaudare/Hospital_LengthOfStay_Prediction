from flask import Flask, url_for, request, jsonify
import pickle
import json
import numpy as np
from sklearn.linear_model import Ridge
import time

app = Flask(__name__)


@app.route('/apitest')
def apitest():
    return "API working"


@app.route('/los', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # print(data)
    # print(model.predict([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 10, 10.4, 138.5897012, 119.6331344, 10, 1.24057185, 28.99494458, 67, 6.3, 1, 4, 5]]))
    prediction = model.predict([np.array(data['data'])])
    output = prediction[0]
    return jsonify(output)


        


if __name__ == "__main__":
  global model
  model = pickle.load(open("LOS_RF_model.pkl", 'rb'))
  app.run(host="0.0.0.0", debug=False, port=5005)
