from flask import Flask, jsonify
from sklearn.externals import joblib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                    help="location of the pickle file")

filename = parser.parse_args().model

app = Flask(__name__)

@app.route('/')
def index():
    return "Stackoverflow Salary Predictor"

@app.route('/sal/<int:x>', methods=['GET'])
def predict(x):
	loaded_model=joblib.load(filename)
	y=loaded_model.predict([[x]])[0]
	sal=jsonify({'salary': round(y,2)})
	return sal

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)
