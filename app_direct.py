#!/usr/bin/env python3
"""
Flask application with direct port binding for Render deployment
"""
import os
import sys
import logging
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory

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

# Set maximum file upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Favicon route
@app.route('/favicon.ico')
def favicon():
    try:
        return send_from_directory(os.path.join(app.root_path, 'static'),
                                  'favicon.ico', mimetype='image/vnd.microsoft.icon')
    except Exception as e:
        logger.error(f"Error serving favicon: {str(e)}")
        return "", 204  # No content response

# Flask routes
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return "<h1>BUMPY</h1><p>Welcome to Bumpy - Your Background Removal Solution!</p>"

# Authentication routes
@app.route('/auth/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # This would normally validate credentials and set session data
        logger.info("Login form submitted")
        return redirect(url_for('dashboard'))
    else:
        try:
            return render_template('login.html')
        except Exception as e:
            logger.error(f"Error rendering login template: {str(e)}")
            return """
            <h1>Login</h1>
            <form action="/auth/login" method="post">
                <div>
                    <label>Email:</label>
                    <input type="email" name="email" required>
                </div>
                <div>
                    <label>Password:</label>
                    <input type="password" name="password" required>
                </div>
                <button type="submit">Login</button>
            </form>
            <p>Don't have an account? <a href="/auth/register">Register</a></p>
            """

@app.route('/auth/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # This would normally create a new user
        logger.info("Registration form submitted")
        return redirect(url_for('login'))
    else:
        try:
            return render_template('register.html')
        except Exception as e:
            logger.error(f"Error rendering register template: {str(e)}")
            return """
            <h1>Register</h1>
            <form action="/auth/register" method="post">
                <div>
                    <label>Name:</label>
                    <input type="text" name="name" required>
                </div>
                <div>
                    <label>Email:</label>
                    <input type="email" name="email" required>
                </div>
                <div>
                    <label>Password:</label>
                    <input type="password" name="password" required>
                </div>
                <div>
                    <label>Confirm Password:</label>
                    <input type="password" name="confirm_password" required>
                </div>
                <button type="submit">Register</button>
            </form>
            <p>Already have an account? <a href="/auth/login">Login</a></p>
            """

@app.route('/auth/logout')
def logout():
    # Clear the session
    session.clear()
    return redirect(url_for('index'))

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

# Background removal routes
@app.route('/bg-remover')
def bg_remover():
    logger.info("Background remover page requested")
    try:
        return render_template('bg_remover.html')
    except Exception as e:
        logger.error(f"Error rendering bg_remover: {str(e)}")
        return "<h1>Background Removal Tool</h1><p>Upload an image to remove its background.</p>"

@app.route('/bg-remover/upload', methods=['POST'])
def upload_image():
    logger.info("Image upload requested")
    
    # Check if the post request has the file part
    if 'image' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({
            "success": False,
            "error": "No file part in the request"
        }), 400
    
    file = request.files['image']
    
    # If user does not select file, browser may also submit an empty part without filename
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({
            "success": False,
            "error": "No selected file"
        }), 400
    
    # Ensure the uploads directory exists
    try:
        # Get user ID from session or use anonymous
        user_id = session.get('user_id', 'anonymous')
        upload_dir = os.path.join('static', 'uploads', user_id)
        results_dir = os.path.join('static', 'results', user_id)
        
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate a unique filename
        original_filename = file.filename
        filename_parts = os.path.splitext(original_filename)
        ext = filename_parts[1].lower()
        unique_filename = f"{uuid.uuid4()}{ext}"
        
        # Save the uploaded file
        file_path = os.path.join(upload_dir, unique_filename)
        file.save(file_path)
        logger.info(f"File saved to {file_path}")
        
        # In a simplified implementation, we'll just return success
        # In reality, you would process the image to remove background
        result_filename = unique_filename
        result_path = os.path.join(results_dir, result_filename)
        
        # Simply copy the file for this demonstration
        # In production, you would use your ML model to process the image
        import shutil
        shutil.copy(file_path, result_path)
        
        # Return the paths for the client to use
        return jsonify({
            "success": True,
            "original_file": f"/static/uploads/{user_id}/{unique_filename}",
            "processed_file": f"/static/results/{user_id}/{result_filename}",
            "message": "Image uploaded successfully. Background removal simulation active."
        })
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error processing upload: {str(e)}"
        }), 500

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