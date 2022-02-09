from flask import Flask
from price_reader import PriceReader
import json

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    print("Make prediction")
    return json.dumps({})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

