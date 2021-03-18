import pandas as pd
import flask
from flask import Flask, jsonify, request
import pickle
import joblib

app = Flask(__name__, template_folder='templates')

filename = "finalized_model.sav"
model = joblib.load(filename)
with open("vectorizer.pickle", "rb") as handle:
	vectorizer = pickle.load(handle)

@app.route('/', methods=['GET', 'POST'])

def predict():
    if request.method == 'GET':
        return(flask.render_template('/main.html'))
    if flask.request.method == 'POST':
        data = request.get_data()
        new=[data]
        message=vectorizer.transform(new)
        pred = model.predict(message)
        pred_msg = "Error"
        if str(pred[0]) == '1':
            pred_msg = "SPAM"
        else:
            pred_msg = "SPAM"
        return flask.render_template('/main.html',
                                     original_input={'TT': data},
                                     result=pred_msg)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
