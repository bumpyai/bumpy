#!/usr/bin/env python3
"""
Simple HTTP server for debugging port binding issues on Render.
Automatically listens on the $PORT environment variable.
"""
import os
import http.server
import socketserver
from http import HTTPStatus
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get port from environment or default to 10000
PORT = int(os.environ.get('PORT', 10000))
logger.info(f"Starting server on port {PORT}")

class DebugHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        logger.info(f"Received GET request on path: {self.path}")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        # Include environment info in response
        env_info = ""
        for key, value in os.environ.items():
            env_info += f"{key}: {value}<br>"
        
        response = f"""
        <html>
        <head><title>Debug Server</title></head>
        <body>
            <h1>Debug Server Running!</h1>
            <p>This is a simple debug server to help diagnose port binding issues on Render.</p>
            <h2>Request Info:</h2>
            <p>Path: {self.path}</p>
            <p>Client: {self.client_address}</p>
            <h2>Environment Variables:</h2>
            <p>{env_info}</p>
        </body>
        </html>
        """
        
        self.wfile.write(response.encode())

def run():
    # Explicitly bind to 0.0.0.0 to listen on all interfaces
    with socketserver.TCPServer(("0.0.0.0", PORT), DebugHandler) as httpd:
        logger.info(f"Server started at http://0.0.0.0:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            httpd.server_close()
            logger.info("Server closed")

if __name__ == "__main__":
    run() 