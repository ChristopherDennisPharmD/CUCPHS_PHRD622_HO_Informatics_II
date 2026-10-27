"""Small local REST terminology service built from a documented CDC teaching subset.

Intentionally exposes an HTTP/JSON boundary so students observe a real API request,
not an internal Python dictionary lookup. No dependencies beyond Python 3.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json

HERE = Path(__file__).resolve().parent
CROSSWALK = json.loads((HERE / "data/teaching_crosswalk.json").read_text(encoding="utf-8"))


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        output = json.dumps(body, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(output)))
        self.end_headers()
        self.wfile.write(output)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            return self.reply(200, {"status": "ok", "service": "Rx-Mix-a-Lot teaching terminology API"})
        prefix = "/api/v1/ndc/"
        if not path.startswith(prefix):
            return self.reply(404, {"error": "Endpoint not found"})
        ndc = unquote(path[len(prefix):])
        item = CROSSWALK["entries"].get(ndc)
        if item is None:
            return self.reply(404, {"ndc11": ndc, "error": "NDC not found in teaching subset"})
        return self.reply(200, {
            "ndc11": ndc,
            "cvx_code": item["cvx_code"],
            "cvx_description": item["cvx_description"],
            "vaccine_group": item["vaccine_group"],
            "package": item["package"],
            "source": CROSSWALK["source"],
        })


def main():
    import argparse
    p = argparse.ArgumentParser(description="Local CDC-derived teaching terminology API")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Teaching API ready at http://127.0.0.1:{args.port}/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
