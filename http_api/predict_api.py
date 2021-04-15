from flask import Blueprint, Response, render_template
from flask import request
import logging
import json
from http_api.predict_model import PredictModel

logger = logging.getLogger('movie_rs')

recommend_api = Blueprint('recommend', __name__, template_folder='../views', static_folder='../views')


@recommend_api.route("/")
def hello_api():
    return recommend_api.send_static_file('index.html')


@recommend_api.route("/GetRecMovies")
def get_rec_movies():
    user_id = int(request.args.get('userId'))
    predict_model = PredictModel()
    rec_movies = predict_model.predict(user_id)
    his_movies = predict_model.get_history_movies(user_id)
    # return_str = str(rec_movies) + '-------------------------------------------------' + str(his_movies)
    return render_template('result.html', his_movies=his_movies, rec_movies=rec_movies)
