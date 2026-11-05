#!/usr/bin/env python3
"""Serve the ePCR Segment Bench dashboard (view only). Stdlib only.

/data.json is read live from disk per request, so a refresh reflects the current results.

  python dashboard/server.py            # http://0.0.0.0:7860
  PORT=8000 python dashboard/server.py
"""
import os, json, http.server, socketserver
import build_data   # same folder

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "7860"))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") in ("/data.json", "/data"):
            try:
                body = json.dumps(build_data.build_payload(), ensure_ascii=False).encode("utf-8")
            except Exception as e:                       # fail loud: surface the real error
                body = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
            else:
                self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def guess_type(self, path):
        t = super().guess_type(path)
        if t in ("text/html", "text/plain", "text/css", "application/javascript", "text/javascript"):
            return t + "; charset=utf-8"
        return t

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")   # always serve the latest
        super().end_headers()

    def log_message(self, fmt, *args):
        print("[server]", self.address_string(), fmt % args)

if __name__ == "__main__":
    os.chdir(HERE)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"[server] ePCR Segment Bench on http://0.0.0.0:{PORT}  (Ctrl-C to stop)")
        print(f"[server] /data.json is built live from dataset/ (input) + results/ (output) per request")
        print(f"[server] if remote, port-forward:  ssh -L {PORT}:$(hostname):{PORT} <user>@<host>")
        httpd.serve_forever()
