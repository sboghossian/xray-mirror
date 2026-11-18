#!/usr/bin/env python3
"""Static dev server for X-ray Mirror. Serves the project root on 127.0.0.1.

Besides static files it accepts POST /__shot/<name>.png and writes the body to
test/out/, so `?test` runs can save a rendered frame for inspection. The page
itself needs nothing but static hosting.
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "test", "out")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        m = re.fullmatch(r"/__shot/([A-Za-z0-9_-]+\.png)", self.path)
        if not m:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, m.group(1)), "wb") as f:
            f.write(body)
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8797
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
