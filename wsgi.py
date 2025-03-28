#!/usr/bin/env python3

# This is a special file that resolves the naming conflict between app.py and app/
# It imports the Flask app instance from our app.py file
import sys
import os
import importlib.util

# Add the current directory to Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

# Pre-download rembg models to avoid timeout issues during requests
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Initializing rembg and pre-downloading models...")

try:
    import rembg
    import rembg.session_factory
    import rembg.sessions.u2net
    import rembg.sessions.isnet
    
    # Create cache directory if specified in environment
    cache_dir = os.environ.get('REMBG_CACHE_DIR', '/tmp/rembg_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Pre-initialize the models to download them
    logging.info("Pre-loading u2net model")
    u2net = rembg.new_session("u2net")
    
    logging.info("Pre-loading isnet model")
    isnet = rembg.new_session("isnet")
    
    logging.info("Models pre-loaded successfully!")
except Exception as e:
    logging.error(f"Error initializing rembg models: {str(e)}")
    logging.info("Background removal may fall back to simpler methods")

# Import the Flask app directly from app.py file, not from the app package
spec = importlib.util.spec_from_file_location("app_module", os.path.join(base_dir, "app.py"))
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)

# This is what Gunicorn will import
app = app_module.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)