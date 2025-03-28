import multiprocessing

# Gunicorn configuration file for Flask app deployment on Render
bind = "0.0.0.0:$PORT"  # Render will supply the PORT environment variable
workers = multiprocessing.cpu_count() * 2 + 1
timeout = 120  # Increased timeout for image processing tasks
max_requests = 1000
max_requests_jitter = 50
preload_app = True
worker_class = "gevent" 