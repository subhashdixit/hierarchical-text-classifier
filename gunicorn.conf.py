"""Bound resources for a CPU-only portfolio service."""

# Gunicorn requires these lowercase setting names.
# pylint: disable=invalid-name
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = 1
threads = 2
timeout = 90
accesslog = "-"
errorlog = "-"
# Do not log URLs/query strings or user-submitted text.
access_log_format = "%(t)s %(m)s %(s)s %(L)s"
