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
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return "<h1>BUMPY</h1><p>Welcome to Bumpy - Your Background Removal Solution!</p>"

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
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        return "<h1>Dashboard</h1><p>Your BUMPY dashboard will appear here.</p>"

@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"Error rendering about: {str(e)}")
        return "<h1>About BUMPY</h1><p>BUMPY is an AI-powered background removal tool.</p>"

@app.route('/pricing')
def pricing():
    try:
        return render_template('pricing.html')
    except Exception as e:
        logger.error(f"Error rendering pricing: {str(e)}")
        return "<h1>BUMPY Pricing</h1><p>Our pricing plans will appear here.</p>"

# Add the missing bg-remover route
@app.route('/bg-remover')
def bg_remover():
    logger.info("Background remover page requested")
    try:
        return render_template('bg_remover.html')
    except Exception as e:
        logger.error(f"Error rendering bg_remover: {str(e)}")
        return "<h1>Background Removal Tool</h1><p>Upload an image to remove its background.</p>"

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error: {request.path}")
    try:
        return render_template('404.html'), 404
    except Exception as template_error:
        logger.error(f"Error rendering 404 template: {str(template_error)}")
        return "<h1>404 - Page Not Found</h1><p>The page you requested could not be found.</p>", 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {str(e)}")
    try:
        return render_template('500.html'), 500
    except Exception as template_error:
        logger.error(f"Error rendering 500 template: {str(template_error)}")
        return "<h1>500 - Server Error</h1><p>An internal server error occurred. Please try again later.</p>", 500

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