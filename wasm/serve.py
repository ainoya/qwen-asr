#!/usr/bin/env python3
"""Static server for the WebAssembly demo.

Two things a plain `python3 -m http.server` will not do:

* Send COOP/COEP so the page is cross-origin isolated. Without that
  SharedArrayBuffer is unavailable and the engine's thread pool cannot start.
* Handle HEAD and Range on a 2 GB file without reading it into memory.

Usage:
    ./wasm/serve.py [--port 8765] [--root .]
Then open  http://localhost:8765/wasm/demo/
"""

import argparse
import http.server
import os
import socketserver
import sys

CHUNK = 1 << 20


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        # CORS, so a playground served from another origin (the GitHub Pages
        # topology) can fetch the model from this server during local tests.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path):
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".bin"):
            return "application/octet-stream"
        return super().guess_type(path)

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()

        size = os.path.getsize(path)
        try:
            units, _, spec = rng.partition("=")
            if units.strip() != "bytes":
                raise ValueError
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except ValueError:
            return super().do_GET()

        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        remaining = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                buf = f.read(min(CHUNK, remaining))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(buf)

    def log_message(self, fmt, *args):
        if "--verbose" in sys.argv:
            super().log_message(fmt, *args)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)

    handler = lambda *a, **kw: Handler(*a, directory=root, **kw)
    with Server(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {root} on http://localhost:{args.port}")
        print(f"open    http://localhost:{args.port}/wasm/demo/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
