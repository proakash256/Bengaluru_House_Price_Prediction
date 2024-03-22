from flask import Flask, Request, jsonify
import util

app = Flask(__name__)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/predict_home_price", methods=["POST"])
def predict_home_price():
    pass

if __name__ == "__main__":
    print("Starting Python Flask Server for Bengaluru Home Price Prediction...")
    app.run()
