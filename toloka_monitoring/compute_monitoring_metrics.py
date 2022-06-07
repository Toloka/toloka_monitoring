import os
import time
import datetime

import numpy as np

from crowdkit.aggregation import MajorityVote
from sklearn.metrics import confusion_matrix

import toloka.client as toloka
from toloka_monitoring.api.db import init_db, db_session_factory
from toloka_monitoring.api.models import Prediction, PredictionLabel, MonitoringCounts
from toloka_monitoring.config import SQLALCHEMY_DATABASE_URI, TOLOKA_API_TOKEN, TOLOKA_PROJECT_ID
from toloka_monitoring.setup_toloka_project import create_pool

def get_predictions_for_labelling(session, limit=10):
    query = session.query(Prediction).join(PredictionLabel, isouter=True)\
        .filter(PredictionLabel.prediction_id.is_(None)) \
        .order_by(Prediction.time_created.desc()) \
        .limit(limit)
    unlabelled_preds = query.all()
    return unlabelled_preds


def wait_pool_for_close(toloka_client, pool_id, minutes_to_wait=0.2):
    sleep_time = 60 * minutes_to_wait
    pool = toloka_client.get_pool(pool_id)
    while not pool.is_closed():
        op = toloka_client.get_analytics([toloka.analytics_request.CompletionPercentagePoolAnalytics(subject_id=pool.id)])
        op = toloka_client.wait_operation(op)
        percentage = op.details['value'][0]['result']['value']
        print(
            f'   {datetime.datetime.now().strftime("%H:%M:%S")}\t'
            f'Pool {pool.id} - {percentage}%'
        )
        time.sleep(sleep_time)
        pool = toloka_client.get_pool(pool.id)
    print('Pool was closed.')


def annotate_with_toloka(predictions, toloka_client, toloka_pool_id):
    pool = toloka_client.get_pool(toloka_pool_id)
    answers_df = toloka_client.get_assignments_df(pool.id)

    tasks = [
        toloka.Task(input_values={'image_url': prediction.image_url, 'pred_id': prediction.id},
                    pool_id=pool.id)
        for prediction in predictions if prediction.id not in answers_df['INPUT:pred_id'].values
    ]
    if tasks:
        toloka_client.create_tasks(tasks, allow_defaults=True)
        pool = toloka_client.open_pool(pool.id)
    wait_pool_for_close(toloka_client, pool.id)

    answers_df = toloka_client.get_assignments_df(pool.id)
    answers_df = answers_df.rename(columns={
        'INPUT:pred_id': 'task',
        'OUTPUT:label': 'label',
        'ASSIGNMENT:worker_id': 'worker',
    })
    aggregated_answers = MajorityVote().fit_predict(answers_df)
    return aggregated_answers.to_dict()


def get_prediction_crowd_annotations(predictions, toloka_client, toloka_pool_id):
    crowd_labels = annotate_with_toloka(predictions, toloka_client, toloka_pool_id)

    crowd_annotations = []
    for prediction in predictions:
        crowd_label = crowd_labels[prediction.id]
        prediction_crowd_label = PredictionLabel(prediction_id=prediction.id, label=crowd_label)
        crowd_annotations.append(prediction_crowd_label)

    return crowd_annotations


def get_monitoring_metrics(predictions, crowd_annotations):
    pred_labels = np.array([max(prediction.prediction.items(), key=lambda r: r[1])[0] for prediction in predictions])
    crowd_labels = np.array([crowd_annotation.label for crowd_annotation in crowd_annotations])
    neither_labels_idx = (crowd_labels == 'neither')
    pred_labels = pred_labels[~neither_labels_idx]
    crowd_labels = crowd_labels[~neither_labels_idx]

    cm = confusion_matrix(crowd_labels, pred_labels)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    return MonitoringCounts(
        tp=int(TP[0]),
        fp=int(FP[0]),
        tn=int(TN[0]),
        fn=int(FN[0]),
        n_neither=len(neither_labels_idx),
    )


def compute_monitoring_metrics():
    db_uri = SQLALCHEMY_DATABASE_URI
    init_db(db_uri)
    session = db_session_factory()
    toloka_client = toloka.TolokaClient(TOLOKA_API_TOKEN, 'PRODUCTION')
    project = toloka_client.get_project(TOLOKA_PROJECT_ID)
    pool = create_pool(toloka_client, project.id)
    predictions = get_predictions_for_labelling(session)
    print(f'Annotating {len(predictions)} predictions with Toloka')
    crowd_annotations = get_prediction_crowd_annotations(predictions, toloka_client, pool.id)
    for prediction_crowd_annotation in crowd_annotations:
        session.add(prediction_crowd_annotation)

    if predictions and crowd_annotations:
        print('Computing monitoring metrics')
        metrics = get_monitoring_metrics(predictions, crowd_annotations)
        session.add(metrics)
    session.commit()


if __name__ == '__main__':
    compute_monitoring_metrics()
