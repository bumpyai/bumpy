import multiprocessing
import os

# Get the port from the environment variable
port = os.environ.get("PORT", "10000")

# Gunicorn configuration file for Flask app deployment on Render
bind = f"0.0.0.0:{port}"  # Explicitly binding to the PORT environment variable
workers = 1  # Reduced to avoid memory issues
timeout = 300  # Increased timeout for image processing tasks (5 minutes)
max_requests = 1000
max_requests_jitter = 50
preload_app = True
worker_class = "gevent"
worker_connections = 1000  # Increase connections for better concurrency
keepalive = 5  # How long to wait for requests on a Keep-Alive connection 