from flask import Blueprint, Response
from flask import request
import logging
import json

logger = logging.getLogger('movie_rs')

recommend_api = Blueprint('recommend', __name__, template_folder='templates', static_folder='../views')


@recommend_api.route("/")
def hello_api():
    return recommend_api.send_static_file('index.html')


