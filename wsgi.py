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
    # Set environment variables for rembg before importing
    os.environ["REMBG_CACHE_DIR"] = os.environ.get('REMBG_CACHE_DIR', '/tmp/rembg_cache')
    os.environ["U2NET_HOME"] = os.environ.get('REMBG_CACHE_DIR', '/tmp/rembg_cache')
    os.environ["REMBG_MODEL_FILENAME"] = "isnet-general-use.pth"  # Use a smaller model by default
    
    # Make sure cache directory exists
    os.makedirs(os.environ["REMBG_CACHE_DIR"], exist_ok=True)
    logging.info(f"Using cache directory: {os.environ['REMBG_CACHE_DIR']}")
    
    # Now import rembg
    import rembg
    
    # Pre-initialize the models to download them
    logging.info("Pre-loading isnet-general-use model (faster)")
    try:
        isnet = rembg.new_session("isnet-general-use")
        logging.info("isnet-general-use model loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading isnet model: {str(e)}")
    
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