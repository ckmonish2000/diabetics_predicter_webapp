from flask import Flask, jsonify, request, render_template
from  models.utils import predict

app = Flask(__name__)


@app.route("/")
def index():
    return "hello"


if __name__ == "__main__":
    app.run(debug=True)
