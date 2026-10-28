"""
Lean HTTP & REST Fact API Server for Solix FactLens.
Zero external dependencies (uses standard library http.server),
sub-millisecond JSON routing, safe frame image streaming, and static asset hosting.
"""

import os
import sys
import json
import mimetypes
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import BASE_DIR, PROCESSED_DIR, FRAMES_DIR
from src.web.fact_engine import FactEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"


class FactLensRequestHandler(SimpleHTTPRequestHandler):
    """
    HTTP request handler supporting REST API endpoints and static asset serving.
    """

    def __init__(self, *args, **kwargs):
        # Cache fact engine instances per session
        self.engines: dict = {}
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def get_engine(self, session_id: str) -> FactEngine:
        """Retrieves or initializes FactEngine for a given session."""
        if not hasattr(self.server, "engines"):
            self.server.engines = {}

        if session_id not in self.server.engines:
            self.server.engines[session_id] = FactEngine(session_id)
        return self.server.engines[session_id]

    def send_json(self, data: Any, status_code: int = 200):
        """Sends JSON response with appropriate headers."""
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        """Handles CORS pre-flight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Dispatches GET requests to API or static handler."""
        parsed = urlparse(self.path)
        path = parsed.path
        query_params = parse_qs(parsed.query)

        # -----------------------------------------------------------------
        # API ROUTING
        # -----------------------------------------------------------------
        if path.startswith("/api/"):
            self.handle_api(path, query_params)
            return

        # -----------------------------------------------------------------
        # FRAME IMAGE SERVING: /frames/<session_id>/<filename>
        # -----------------------------------------------------------------
        if path.startswith("/frames/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 3:
                session_id = parts[1]
                frame_filename = parts[2]
                frame_path = FRAMES_DIR / session_id / frame_filename
                if frame_path.exists() and frame_path.is_file():
                    self.serve_file(frame_path, "image/jpeg")
                    return
            self.send_error(404, "Frame image not found")
            return

        # -----------------------------------------------------------------
        # STATIC ASSETS
        # -----------------------------------------------------------------
        if path == "/" or path == "":
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                self.serve_file(index_path, "text/html; charset=utf-8")
                return

        file_path = STATIC_DIR / path.lstrip("/")
        if file_path.exists() and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            self.serve_file(file_path, mime_type or "application/octet-stream")
            return

        # Fallback to index.html for single-page routing
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            self.serve_file(index_path, "text/html; charset=utf-8")
            return

        self.send_error(404, f"File not found: {path}")

    def handle_api(self, path: str, query_params: dict):
        """Processes REST API routes."""
        parts = [p for p in path.strip("/").split("/") if p]
        # parts: ['api', ...]

        # GET /api/sessions
        if path == "/api/sessions":
            sessions = []
            if PROCESSED_DIR.exists():
                for p in PROCESSED_DIR.iterdir():
                    if p.is_dir() and (p / "timeline_manifest.json").exists():
                        sessions.append(p.name)
            # Ensure solix_demo_test is prioritized if present
            if "solix_demo_test" in sessions:
                sessions.remove("solix_demo_test")
                sessions.insert(0, "solix_demo_test")
            self.send_json({"sessions": sessions})
            return

        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "sessions":
            session_id = parts[2]
            engine = self.get_engine(session_id)

            # GET /api/sessions/<session_id>/summary
            if len(parts) == 4 and parts[3] == "summary":
                self.send_json(engine.get_summary_facts())
                return

            # GET /api/sessions/<session_id>/facts
            if len(parts) == 4 and parts[3] == "facts":
                category = query_params.get("category", ["all"])[0]
                if category == "bottlenecks":
                    self.send_json({"facts": engine.get_friction_facts()})
                elif category == "automations":
                    self.send_json({"facts": engine.get_automation_facts()})
                elif category == "flow":
                    self.send_json({"facts": engine.get_step_facts()})
                elif category == "time":
                    self.send_json({"facts": engine.get_time_facts()})
                else:
                    self.send_json({
                        "summary": engine.get_summary_facts(),
                        "friction": engine.get_friction_facts(),
                        "automations": engine.get_automation_facts(),
                        "steps": engine.get_step_facts(),
                        "time_allocation": engine.get_time_facts(),
                    })
                return

            # GET /api/sessions/<session_id>/query?q=...
            if len(parts) == 4 and parts[3] == "query":
                q = query_params.get("q", [""])[0]
                results = engine.query_facts(q)
                self.send_json({
                    "query": q,
                    "count": len(results),
                    "results": results,
                })
                return

            # GET /api/sessions/<session_id>/slot/<timestamp>
            if len(parts) == 5 and parts[3] == "slot":
                try:
                    ts = int(parts[4])
                    slot = engine.get_timeline_slot(ts)
                    if slot:
                        self.send_json(slot)
                        return
                except ValueError:
                    pass
                self.send_error(404, "Slot timestamp not found")
                return

        # GET /api/automations/<script_name>
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "automations":
            script_name = parts[2]
            if not script_name.endswith(".py"):
                script_name += ".py"
            script_path = BASE_DIR / "src" / "automations" / script_name
            if script_path.exists() and script_path.is_file():
                self.send_json({
                    "filename": script_name,
                    "code": script_path.read_text(encoding="utf-8"),
                })
                return
            self.send_error(404, f"Automation script {script_name} not found")
            return

        self.send_error(404, f"API endpoint not found: {path}")

    def serve_file(self, file_path: Path, mime_type: str):
        """Streams a local file with proper headers."""
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")


def run_server(port: int = 8000, host: str = "0.0.0.0"):
    """Starts the Solix FactLens web server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, FactLensRequestHandler)
    print(f"\n==================================================================")
    print(f" SOLIX FACTLENS - LEAN WORKFLOW FACT ENGINE")
    print(f" Web Interface: http://localhost:{port}")
    print(f" Mode: Lean, Permissive, Zero-Bloat (No Emojis)")
    print(f" Press Ctrl+C to stop.")
    print(f"==================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Solix FactLens Web Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    run_server(port=args.port, host=args.host)

