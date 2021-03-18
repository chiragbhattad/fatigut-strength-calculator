import flask
import pickle
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')

model1 = pickle.load(open("model1.pkl","rb"))
model1._make_predict_function()
model2 = pickle.load(open("model2.pkl","rb"))
model3 = pickle.load(open("model3.pkl","rb"))
model4 = pickle.load(open("model4.pkl","rb"))
model5 = pickle.load(open("model5.pkl","rb"))

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('/index1.html'))
    if flask.request.method == 'POST':
        TT = flask.request.form['TT']
        C = flask.request.form['C']
        Cr = flask.request.form['Cr']
        Mn = flask.request.form['Mn']
        P = flask.request.form['P']
        input_variables = pd.DataFrame([[TT, C, Cr, Mn, P]], columns=['TT', 'C', 'Cr', 'Mn', 'P'], dtype=float)
        pred = (model1.predict(input_variables)[0] + model2.predict(input_variables)[0] + model3.predict(input_variables)[0] + model4.predict(input_variables)[0] + model5.predict(input_variables)[0])/5

        return flask.render_template('/index1.html',
                                     original_input={'TT': TT,
                                                     'C': C,
                                                     'Cr': Cr,
                                                     'Mn': Mn,
                                                     'P': P},
                                     result=pred,)

if __name__ == '__main__':
    app.run()
