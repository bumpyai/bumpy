#!/usr/bin/env python3
"""
Ultra-simple HTTP server for Render deployment.
"""
import os
import http.server
import socketserver
import socket
import sys

# Print debug info
print("=" * 50)
print("SIMPLE SERVER STARTING")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
print("Environment variables:")
for key, value in os.environ.items():
    print(f"  {key}: {value}")
print("=" * 50)

# Get PORT from environment or use 10000 as default
PORT = int(os.environ.get('PORT', 10000))
print(f"Starting server on port {PORT}")

# Simple request handler
class SimpleHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        response = f"""
        <html>
        <head><title>Simple Server</title></head>
        <body>
            <h1>Server is running!</h1>
            <p>Port: {PORT}</p>
            <p>Path: {self.path}</p>
            <p>Server hostname: {socket.gethostname()}</p>
        </body>
        </html>
        """
        
        self.wfile.write(response.encode())
        print(f"Handled request for {self.path}")

# Create and run the server
httpd = socketserver.TCPServer(("", PORT), SimpleHandler)
print(f"Server started at http://0.0.0.0:{PORT}")
print(f"Server socket: {httpd.socket}")
print(f"Server socket name: {httpd.socket.getsockname()}")
httpd.serve_forever() 