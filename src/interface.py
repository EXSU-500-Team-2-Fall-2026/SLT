"""Local dermoscopy upload demo. Run with python -m src.interface."""
import hashlib
import json
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000


def load_study_model():
    reports = ROOT / "reports/v2"
    frozen = json.loads((reports / "final/run_record.json").read_text())["frozen"]["cnn"]
    checkpoint = ROOT / frozen["model_file"]
    if not checkpoint.is_file():
        return None, "Scoring unavailable. The study checkpoint models/cnn.pt is missing."
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != frozen["model_sha256"]:
        return None, "Scoring unavailable. The checkpoint differs from the evaluated CNN."
    from src.predict import load_models

    loaded = load_models(checkpoint.parent, reports, names=("cnn",))
    if "cnn" not in loaded["models"]:
        return None, "Scoring unavailable. Check the Python model dependencies."
    loaded["calibrators"]["cnn"] = frozen["calibrator"]
    loaded["thresholds"]["cnn"] = frozen["threshold"]
    return loaded, None


class ReviewHandler(BaseHTTPRequestHandler):
    def respond(self, status, body, content_type="application/json"):
        if content_type == "application/json":
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.respond(200, (ROOT / "interface/index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/status":
            self.respond(200, {"available": self.server.model is not None, "message": self.server.problem})
        else:
            self.respond(404, {"error": "Page unavailable."})

    def do_POST(self):
        if self.path != "/predict":
            return self.respond(404, {"error": "Unknown endpoint."})
        if self.headers.get_content_type() not in ("image/jpeg", "image/png"):
            return self.respond(415, {"error": "Choose a JPEG or PNG image."})
        if self.server.model is None:
            return self.respond(503, {"error": self.server.problem})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BYTES:
            return self.respond(413, {"error": "Choose a nonempty image up to 10 MiB."})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image"
            data = self.rfile.read(length)
            if len(data) != length:
                return self.respond(400, {"error": "Incomplete upload."})
            path.write_bytes(data)
            try:
                with Image.open(path, formats=["JPEG", "PNG"]) as image:
                    if image.width * image.height > MAX_PIXELS or image.n_frames != 1:
                        return self.respond(422, {"error": "Use a single image up to 16 megapixels."})
                    image.load()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                return self.respond(422, {"error": "Unable to read this image."})
            try:
                from src.predict import predict_image
                result = predict_image(path, self.server.model)["cnn"]
                self.respond(200, result)
            except Exception:
                self.respond(500, {"error": "Unable to score this image."})


if __name__ == "__main__":
    with HTTPServer(("127.0.0.1", 8000), ReviewHandler) as server:
        server.model, server.problem = load_study_model()
        print(server.problem or "Study CNN loaded.", flush=True)
        print("Open http://127.0.0.1:8000", flush=True)
        server.serve_forever()
