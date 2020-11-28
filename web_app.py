import flask
from flask import Flask, request
from joblib import load, dump
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route("/")
def hello_world():
    svc = load('svc.joblib')
    cnn = load_model('cnn.h5')
    data = []
    data_exists = request.args.get('0,0') is not None
    if data_exists:
        # collect all data
        for rows in range(28):
            for cols in range(28):
                # print('indeks je', str(index)+str(index))
                data.append(request.args.get(str(rows)+','+str(cols)))

    if data_exists:
        return flask.render_template('view.html', svc_num=svc.predict(np.reshape(np.array(data), (-1, 784))),
                                     cnn_num=np.argmax(cnn.predict(np.reshape(np.array(data), (-1, 28, 28, 1)).astype(np.float)), axis=1))
    else:
        return flask.render_template('view.html')
