from flask import Flask, jsonify
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def home():
    logger.info("Received request on root endpoint")
    return jsonify({
        "status": "ok",
        "message": "Test app is running"
    })

@app.route('/health')
def health():
    logger.info("Received request on health endpoint")
    return jsonify({
        "status": "healthy",
        "environment": os.environ.get("FLASK_ENV", "unknown")
    })

if __name__ == "__main__":
    # Get PORT from environment variable or use 10000 as default
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Flask app on port {port}")
    
    # Explicitly print this to logs for debugging
    print(f"STARTING FLASK ON PORT {port}")
    
    # Run the app on 0.0.0.0 (all interfaces) and the specified port
    app.run(host="0.0.0.0", port=port, debug=True) 