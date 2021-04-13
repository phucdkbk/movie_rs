from flask import Flask
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import CORS
from http_api.predict_api import recommend_api


class FlaskConfig:
    host = '127.0.0.1'
    port = 5000


def init_logger():
    logger = logging.getLogger('movie_rs')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    rotate_handler = RotatingFileHandler('../logs/movie_rs.log', maxBytes=200000000, backupCount=10)
    rotate_handler.setLevel(logging.INFO)
    rotate_handler.setFormatter(formatter)

    logger.addHandler(rotate_handler)
    logger.addHandler(ch)


def start_api():
    app = Flask(__name__, static_folder='../views')
    app.secret_key = 'super secret key'
    CORS(app, max_age=86400)
    app.register_blueprint(recommend_api)
    app.debug = True
    app.run(host=FlaskConfig.host, port=FlaskConfig.port, threaded=True)


if __name__ == "__main__":
    init_logger()
    start_api()
