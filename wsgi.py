"""
WSGI entry point for the BUMPY application
"""

# This is a workaround for the module name conflict issue
import sys
import os

# Get the current directory and add it to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the Flask application using the full file name
import app
application = app.app

# For compatibility with common WSGI servers
app = application

if __name__ == "__main__":
    application.run() 