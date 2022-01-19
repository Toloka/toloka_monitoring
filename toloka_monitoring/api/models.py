import uuid

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from .db import Base

def default_uuid4():
    return str(uuid.uuid4())

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(String, primary_key=True, default=default_uuid4)
    time_created = Column(DateTime(), server_default=func.now())
    image_url = Column(String())
    prediction = Column(JSON())


class PredictionLabel(Base):
    __tablename__ = 'prediction_labels'
    id = Column(String, primary_key=True, default=default_uuid4)
    time_created = Column(DateTime(), server_default=func.now())
    prediction_id = Column(Integer, ForeignKey('predictions.id'))
    label = Column(String())

    prediction = relationship("Prediction", backref=backref("labels", uselist=False), lazy='joined')


class MonitoringCounts(Base):
    __tablename__ = 'monitoring_metrics'
    id = Column(String, primary_key=True, default=default_uuid4)
    time_created = Column(DateTime(), server_default=func.now())

    tp = Column(Integer())
    fp = Column(Integer())
    tn = Column(Integer())
    fn = Column(Integer())
    n_neither = Column(Integer())


