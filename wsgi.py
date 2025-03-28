#!/usr/bin/env python3

# This is a special file that resolves the naming conflict between app.py and app/
# It imports the Flask app instance from our app.py file
import sys
import os

# Add the current directory to Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

# Import the Flask app from app.py
from app import app as flask_app

# This is what Gunicorn will import
app = flask_app

if __name__ == "__main__":
    app.run() 