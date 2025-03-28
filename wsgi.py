#!/usr/bin/env python3

# This is a special file that resolves the naming conflict between app.py and app/
# It imports the Flask app instance from our app.py file
import sys
import os
import importlib.util

# Add the current directory to Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

# Import the Flask app directly from app.py file, not from the app package
spec = importlib.util.spec_from_file_location("app_module", os.path.join(base_dir, "app.py"))
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)

# This is what Gunicorn will import
app = app_module.app

if __name__ == "__main__":
    app.run() 