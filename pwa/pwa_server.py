"""
pwa_server.py
─────────────
Lightweight static HTTP server that serves the PWA shell files
on http://localhost:8502 — completely offline, no internet needed.

This server runs alongside Streamlit (port 8501) and provides:
  • /index.html   → The PWA wrapper page
  • /manifest.json
  • /sw.js        → Service Worker
  • /icon-192.png
  • /icon-512.png

Run with:  python pwa_server.py
"""

import http.server
import socketserver
import os
import sys
import threading
import webbrowser
import time

PWA_DIR  = os.path.dirname(os.path.abspath(__file__))
HOST     = "localhost"
PORT     = 8502


class PWAHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files from the pwa/ directory with correct MIME types."""

    # Map extension → Content-Type
    EXTRA_MIME = {
        ".webmanifest": "application/manifest+json",
        ".json":        "application/json",
        ".js":          "application/javascript",
        ".png":         "image/png",
        ".html":        "text/html; charset=utf-8",
        ".ico":         "image/x-icon",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PWA_DIR, **kwargs)

    def end_headers(self):
        # Required headers for Service Workers to work on localhost
        self.send_header("Service-Worker-Allowed", "/")
        self.send_header("Cache-Control", "no-cache")
        # Allow iframe embedding from localhost (needed for PWA shell ↔ Streamlit)
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def guess_type(self, path):
        _, ext = os.path.splitext(path)
        if ext in self.EXTRA_MIME:
            return self.EXTRA_MIME[ext]
        return super().guess_type(path)

    def do_GET(self):
        # Redirect root to index.html
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, fmt, *args):
        # Quiet logging — only print non-200 responses
        code = args[1] if len(args) > 1 else ""
        if code not in ("200", "304"):
            super().log_message(fmt, *args)


def open_browser_delayed(url: str, delay: float = 1.5):
    """Open the browser after a short delay so the server is ready."""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    os.chdir(PWA_DIR)

    print("=" * 58)
    print("  🩺  AI Malaria Lab Assistant — PWA Server")
    print("=" * 58)
    print(f"\n  PWA Shell  ➜  http://{HOST}:{PORT}")
    print(f"  Streamlit  ➜  http://{HOST}:8501  (must be running)")
    print("\n  Open http://localhost:8502 in Chrome or Edge.")
    print("  Look for the ⬇  Install button in the address bar.\n")
    print("  Press Ctrl+C to stop.\n")
    print("=" * 58)

    # Allow port reuse so quick restarts work
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer((HOST, PORT), PWAHandler) as httpd:
        open_browser_delayed(f"http://{HOST}:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n  PWA server stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
