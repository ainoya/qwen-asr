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
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()

        size = os.path.getsize(path)
        rng = self.headers.get("Range")

        if not rng:
            if size > CHUNK:
                self.send_response(200)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                remaining = size
                last_logged = 0
                with open(path, "rb") as f:
                    while remaining > 0:
                        buf = f.read(min(CHUNK, remaining))
                        if not buf:
                            break
                        try:
                            self.wfile.write(buf)
                        except Exception as e:
                            sent = size - remaining
                            print(f"[STREAM] Client disconnected after {sent / 1e6:.1f} MB / {size / 1e6:.1f} MB ({sent/size*100:.1f}%): {e}", flush=True)
                            return
                        remaining -= len(buf)
                        sent = size - remaining
                        if sent - last_logged >= 100 * 1024 * 1024 or remaining == 0:
                            last_logged = sent
                            print(f"[STREAM] Sent {sent / 1e6:.1f} / {size / 1e6:.1f} MB ({sent/size*100:.1f}%)", flush=True)
                return
            return super().do_GET()

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
        total_range = remaining
        last_logged = 0
        with open(path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                buf = f.read(min(CHUNK, remaining))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except Exception as e:
                    sent = total_range - remaining
                    print(f"[RANGE] Client disconnected after {sent / 1e6:.1f} MB / {total_range / 1e6:.1f} MB ({sent/total_range*100:.1f}%): {e}", flush=True)
                    return
                remaining -= len(buf)
                sent = total_range - remaining
                if sent - last_logged >= 100 * 1024 * 1024 or remaining == 0:
                    last_logged = sent
                    print(f"[RANGE] Sent {sent / 1e6:.1f} / {total_range / 1e6:.1f} MB ({sent/total_range*100:.1f}%)", flush=True)

    def log_message(self, fmt, *args):
        if "--verbose" in sys.argv:
            super().log_message(fmt, *args)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ssl", action="store_true", help="Enable HTTPS using SSL certificates")
    ap.add_argument("--cert", default="wasm/certs/cert.pem", help="Path to SSL certificate")
    ap.add_argument("--key", default="wasm/certs/key.pem", help="Path to SSL private key")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)

    handler = lambda *a, **kw: Handler(*a, directory=root, **kw)
    with Server((args.host, args.port), handler) as httpd:
        proto = "http"
        cert_path = os.path.join(root, args.cert) if not os.path.isabs(args.cert) else args.cert
        key_path = os.path.join(root, args.key) if not os.path.isabs(args.key) else args.key
        if args.ssl or (os.path.exists(cert_path) and os.path.exists(key_path) and args.ssl):
            import ssl
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
            proto = "https"

        print(f"serving {root} on {proto}://{args.host}:{args.port}")
        print(f"local:  {proto}://localhost:{args.port}/wasm/demo/")
        if args.host in ("0.0.0.0", ""):
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(0.2)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                print(f"LAN:    {proto}://{local_ip}:{args.port}/wasm/demo/")
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
