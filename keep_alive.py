import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = 8080

def run_server():
    server_address = ("0.0.0.0", PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    httpd.serve_forever()

def keep_alive():
    thread = threading.Thread(target=run_server)
    thread.daemon = True
    thread.start()
    print(f"✅ Keep-Alive aktif: Port {PORT}")
