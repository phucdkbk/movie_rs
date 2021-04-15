from flask import Flask
from flask_cors import CORS
from http_api.predict_api import recommend_api
from config import FlaskConfig


def start_api():
    app = Flask(__name__, static_folder='../views')
    app.secret_key = 'super secret key'
    CORS(app, max_age=86400)
    app.register_blueprint(recommend_api)
    app.debug = True
    app.run(host=FlaskConfig.host, port=FlaskConfig.port, threaded=True)


if __name__ == "__main__":
    start_api()
