from flask import g
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = None
session_maker = sessionmaker(autocommit=False, autoflush=False)
db_session_factory = scoped_session(session_maker)
Base = declarative_base()


def init_db(db_uri):
    global engine
    engine = create_engine(db_uri, convert_unicode=True)
    session_maker.configure(bind=engine)

    from . import models
    Base.metadata.create_all(bind=engine)


def get_session():
    if not hasattr(g, 'db_session'):
        g.db_session = db_session_factory()
    return g.db_session
