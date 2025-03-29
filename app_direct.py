#!/usr/bin/env python3
"""
Flask application with direct port binding for Render deployment
"""
import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print debug info
logger.info("=" * 50)
logger.info("FLASK APP STARTING WITH DIRECT BINDING")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info("=" * 50)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "env": os.environ.get('FLASK_ENV', 'production')
    }), 200

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Create required directories
def create_required_directories():
    """Create necessary directories for file storage"""
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    
    # Create directories for anonymous and dev users
    os.makedirs('static/uploads/anonymous', exist_ok=True)
    os.makedirs('static/results/anonymous', exist_ok=True)
    os.makedirs('static/uploads/dev-user-123', exist_ok=True)
    os.makedirs('static/results/dev-user-123', exist_ok=True)

# Run the app directly
if __name__ == "__main__":
    # Create required directories
    create_required_directories()
    
    # Get PORT from environment or use 10000 as default
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting Flask app on port {port}")
    
    # Run the app on 0.0.0.0 (all interfaces) and the specified port
    app.run(host="0.0.0.0", port=port, threaded=True) 