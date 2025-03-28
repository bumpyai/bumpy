from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import os
import json
from app.auth import auth_bp
from app.bg_remover import bg_remover_bp
import firebase_admin
from firebase_admin import credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment setting
flask_env = os.environ.get('FLASK_ENV', 'production')
is_development = flask_env == 'development'

if is_development:
    print(f"Running in development mode")
else:
    print(f"Running in production mode")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')

# Configure app
app.config.update(
    UPLOAD_FOLDER='static/uploads',
    DEVELOPMENT=is_development
)

# Configure CORS
if is_development:
    # In development, allow all localhost origins
    CORS(app, resources={r"/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000"]}})
else:
    # In production, restrict to the actual domain
    CORS(app, resources={r"/*": {"origins": os.environ.get('ALLOWED_ORIGINS', '*')}})

# Initialize Firebase Admin SDK if credentials are available
try:
    # Try to use firebase-key.json first
    if os.path.exists('firebase-key.json'):
        try:
            cred = credentials.Certificate('firebase-key.json')
            firebase_admin.initialize_app(cred)
            print("Firebase initialized with service account")
        except Exception as e:
            print(f"Warning: Could not initialize Firebase with service account: {e}")
            if is_development:
                # Skip Firebase initialization in development if credentials are invalid
                print("Running in development mode without Firebase Admin SDK")
            else:
                raise
    # Fallback to environment variables
    else:
        # Create a credential dict from environment variables
        firebase_cred = {
            "type": os.environ.get("FIREBASE_ADMIN_TYPE", "service_account"),
            "project_id": os.environ.get("FIREBASE_ADMIN_PROJECT_ID"),
            "private_key_id": os.environ.get("FIREBASE_ADMIN_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("FIREBASE_ADMIN_PRIVATE_KEY").replace("\\n", "\n") if os.environ.get("FIREBASE_ADMIN_PRIVATE_KEY") else None,
            "client_email": os.environ.get("FIREBASE_ADMIN_CLIENT_EMAIL"),
            "client_id": os.environ.get("FIREBASE_ADMIN_CLIENT_ID"),
            "auth_uri": os.environ.get("FIREBASE_ADMIN_AUTH_URI"),
            "token_uri": os.environ.get("FIREBASE_ADMIN_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.environ.get("FIREBASE_ADMIN_AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.environ.get("FIREBASE_ADMIN_CLIENT_X509_CERT_URL")
        }
        # Only initialize Firebase if we have valid credentials
        if firebase_cred["project_id"] and firebase_cred["private_key"]:
            cred = credentials.Certificate(firebase_cred)
            firebase_admin.initialize_app(cred)
        else:
            print("WARNING: Firebase credentials not found. Authentication features will not work properly.")
except Exception as e:
    print(f"WARNING: Failed to initialize Firebase: {e}")
    print("Authentication features will not work properly.")

# Context processor to make Firebase config available to all templates
@app.context_processor
def inject_firebase_config():
    # Use the same Firebase configuration for both development and production
    return {
        'firebase_api_key': os.environ.get('FIREBASE_API_KEY', 'AIzaSyDBXz-8D9pKWIUTC9InEbftAxrtDwquw0Q'),
        'firebase_auth_domain': os.environ.get('FIREBASE_AUTH_DOMAIN', 'bumpy-52866.firebaseapp.com'),
        'firebase_project_id': os.environ.get('FIREBASE_PROJECT_ID', 'bumpy-52866'),
        'firebase_storage_bucket': os.environ.get('FIREBASE_STORAGE_BUCKET', 'bumpy-52866.firebasestorage.app'),
        'firebase_messaging_sender_id': os.environ.get('FIREBASE_MESSAGING_SENDER_ID', '80805253315'),
        'firebase_app_id': os.environ.get('FIREBASE_APP_ID', '1:80805253315:web:b8ac160e359104a1591276'),
        'firebase_measurement_id': os.environ.get('FIREBASE_MEASUREMENT_ID', 'G-T0M9JPFEF6')
    }

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(bg_remover_bp)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # In development mode, allow access to dashboard without authentication
    if is_development and 'user_id' not in session:
        session['user_id'] = 'dev-user-123'
        session['email'] = 'dev@example.com'
        
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = "healthy"
    message = "Service is running normally"
    
    # Check if rembg is working
    try:
        import rembg
        # Just import to verify it's available
        status_details = {"rembg": "available"}
    except Exception as e:
        status = "degraded"
        message = f"rembg not available: {str(e)}"
        status_details = {"rembg": "unavailable", "error": str(e)}
    
    # Check directories
    upload_dir = os.path.join('static', 'uploads')
    results_dir = os.path.join('static', 'results')
    
    dirs_ok = os.path.exists(upload_dir) and os.path.exists(results_dir)
    if not dirs_ok:
        status = "degraded"
        message = "Storage directories not available"
        status_details["directories"] = "missing"
    
    return jsonify({
        "status": status,
        "version": "1.0.0",
        "environment": flask_env,
        "message": message,
        "details": status_details
    }), 200 if status == "healthy" else 207

@app.route('/settings')
def settings():
    # In development mode, allow access to settings without authentication
    if is_development and 'user_id' not in session:
        session['user_id'] = 'dev-user-123'
        session['email'] = 'dev@example.com'
        
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('settings.html')

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
    
    # Create directory for Firebase emulator in development
    if is_development:
        os.makedirs('database', exist_ok=True)

# Setup function to be called when the app is ready
def setup():
    create_required_directories()

# Call setup function when in development
if is_development:
    setup()

if __name__ == '__main__':
    # In development, create directories and run the app with debug mode
    setup()
    
    # Run with host set to 0.0.0.0 to allow external connections
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=is_development) 