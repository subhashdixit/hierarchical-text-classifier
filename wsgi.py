"""Production entry point: gunicorn --config gunicorn.conf.py wsgi:app."""

from hierarchical_classifier.web import create_app

app = create_app()
