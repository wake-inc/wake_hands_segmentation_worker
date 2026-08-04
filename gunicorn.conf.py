"""Gunicorn configuration for one persistent GPU worker process."""

import os

wsgi_app = "care_ego.http:create_app()"
bind = os.environ.get("WAKE_BIND", "0.0.0.0:8080")

# A second process would duplicate the large model in GPU memory. Threads keep
# multiple long-poll/SSE sockets open while requests wait on the internal queue.
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("WAKE_HTTP_THREADS", "8"))
preload_app = False

# Active blocking and SSE requests may run for arbitrarily long videos.
timeout = 0
graceful_timeout = 120
keepalive = 5

# The Gunicorn master automatically starts a replacement if the worker exits.
max_requests = 0
accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = os.environ.get("WAKE_LOG_LEVEL", "info")
