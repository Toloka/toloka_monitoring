import os
import requests

from flask import Flask, render_template
from flask import g
from flask_restx import Api, Resource, fields
from PIL import Image
import pandas as pd
from io import BytesIO

from .db import init_db, get_session
from ..ml.model import ImageClassifier
from ..ml.data import transforms
from .models import Prediction, MonitoringCounts
from toloka_monitoring.config import SQLALCHEMY_DATABASE_URI


def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)
    return d


def create_app():
    model = ImageClassifier.load_from_checkpoint('models/model.ckpt')
    model.eval()

    app = Flask(__name__, static_url_path="/static", static_folder='static')
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI

    init_db(app.config['SQLALCHEMY_DATABASE_URI'])

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        if hasattr(g, 'db_session'):
            g.db_session.close()

    api = Api(app, version='1.0', title='ML API Example', validate=True)

    model_namespace = api.namespace('model', description='Cats vs Dogs model')

    input_row = api.model('Input', {
        'image_url': fields.Url(absolute=True),
    })

    prediction_row = api.model('PredictonRow', {
        'cat': fields.Float(),
        'dog': fields.Float(),
    })

    prediction = api.model('Prediction', {
        'id': fields.String(),
        'predicted_probas': fields.Nested(prediction_row),
    })

    @model_namespace.route('/')
    class ClassifierResource(Resource):
        @model_namespace.doc('predict')
        @model_namespace.expect(input_row)
        @model_namespace.marshal_with(prediction, code=200)
        def post(self):
            session = get_session()
            payload = api.payload
            # fetch img
            image_url = payload['image_url']

            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))

            labels = ['cat', 'dog']
            predicted_probas = model.predict_proba(transforms(img).unsqueeze(0))[0].tolist()
            predicted_probas = {l: round(predicted_probas[ix], 4) for ix, l in enumerate(labels)}

            # write to db
            predicted = Prediction(image_url=image_url, prediction=predicted_probas)
            session.add(predicted)
            session.commit()

            output_json = {
                'id': predicted.id,
                'predicted_probas': predicted.prediction,
            }
            return output_json


    @app.route('/monitoring')
    def monitoring():
        session = get_session()
        metrics = session.query(MonitoringCounts).order_by(MonitoringCounts.time_created.asc()).all()

        metrics_json = {}
        if metrics:
            rows = [row2dict(m) for m in metrics]
            print(rows)
            df = pd.DataFrame.from_records(rows)
            df['timestamp'] = df['time_created'].apply(lambda x: x.timestamp())
            df['precision'] = df['tp']/(df['tp'] + df['fp'])
            df['recall'] = df['tp'] /(df['tp'] + df['fn'])
            df['f1_score'] = 2*(df['precision'] * df['recall'])/(df['precision'] + df['recall'])
            df['time_created_str'] = df['time_created'].apply(lambda dt: dt.strftime('%d/%m/%y %H:%M'))
            metrics_json = df.to_dict('list')

        return render_template('monitoring.html', metrics=metrics_json)

    return app
