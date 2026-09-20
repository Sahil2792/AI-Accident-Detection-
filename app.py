"""Application entry point for the AI Accident Detection project."""

import mimetypes
import os
import torch

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

from urllib.parse import unquote, urlparse
from flask import Flask, Response, g, send_file, session

from werkzeug.utils import secure_filename

from config import Config
from database.init_db import initialize_database, seed_demo_data
from database.init_db import get_connection
from routes.main import main_bp

# Only real media/report files may be downloaded. This blocks serving source
# code, the SQLite database, or any other sensitive file even if a crafted
# path sneaks past the directory checks.
ALLOWED_DOWNLOAD_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm",
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp",
    ".pdf", ".csv",
}


def _safe_download_bases(app: Flask) -> list:
    """Return the realpath'd list of directories downloads may come from.

    Covers every location this app stores media: static uploads/results/
    screenshots, project-root uploads (webcam recordings), results/,
    recordings/, reports/, and the project root itself.
    """
    root = os.path.abspath(app.root_path)
    candidates = [
        root,
        os.path.join(root, "static"),
        os.path.join(root, "static", "uploads"),
        os.path.join(root, "static", "results"),
        os.path.join(root, "static", "screenshots"),
        os.path.join(root, "uploads"),
        os.path.join(root, "uploads", "recordings"),
        os.path.join(root, "recordings"),
        os.path.join(root, "reports"),
    ]

    bases: list = []
    seen = set()
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if os.path.isdir(real) and real not in seen:
            seen.add(real)
            bases.append(real)
    return bases


def _is_relative_to(path: str, base: str) -> bool:
    """True when ``path`` lives inside ``base`` (traversal-safe)."""
    try:
        return os.path.commonpath([path, base]) == base
    except ValueError:
        return False


def _normalize_stored_path(raw_filename: str) -> str | None:
    """Clean a DB-stored path into a plain relative-or-absolute filesystem path."""
    if not raw_filename:
        return None

    cleaned = str(raw_filename).replace("\\", "/").strip()
    if not cleaned:
        return None

    # Some legacy rows may contain full URLs ("/static/uploads/x.jpg" or an
    # absolute http URL) - unwrap them down to the path portion.
    parsed = urlparse(cleaned)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        cleaned = unquote(parsed.path)
    else:
        cleaned = unquote(cleaned)

    for prefix in ("/static/", "static/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    cleaned = cleaned.strip("/")
    return cleaned or None


def _resolve_download_file(app: Flask, raw_filename):
    """Locate a requested file on disk across all safe base directories.

    Accepts absolute paths and relative filenames exactly as stored in the
    database ("uploads/x.jpg", "results/detected_x.mp4",
    "uploads/recordings/user_1_x.mp4", "screenshots/accident_x.jpg", ...).

    Returns the verified absolute path, or ``None`` when the file cannot be
    found inside any safe base directory.
    """
    cleaned = _normalize_stored_path(raw_filename)
    if not cleaned:
        return None

    # Extension whitelist - second line of defence against traversal abuse.
    extension = os.path.splitext(cleaned)[1].lower()
    if extension not in ALLOWED_DOWNLOAD_EXTENSIONS:
        return None

    bases = _safe_download_bases(app)

    if os.path.isabs(cleaned):
        candidates = [cleaned]
    else:
        candidates = [os.path.join(base, cleaned) for base in bases]

    for candidate in candidates:
        real = os.path.realpath(candidate)
        if not os.path.isfile(real):
            continue
        # Traversal guard: the resolved file MUST sit inside a safe base dir.
        if any(_is_relative_to(real, base) for base in bases):
            return real
    return None


def _plain_text_response(message: str, status: int) -> Response:
    """Plain-text error body so browsers never save an HTML page as .htm."""
    return Response(message + "\n", status=status, mimetype="text/plain")


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

    initialize_database(app.config["DATABASE_PATH"])
    seed_demo_data(app.config["DATABASE_PATH"])

    app.register_blueprint(main_bp)

    @app.before_request
    def load_current_user() -> None:
        """Load the signed-in user into request state for templates and guards."""
        user_id = session.get("user_id")
        if not user_id:
            g.current_user = None
            return

        conn = get_connection(app.config["DATABASE_PATH"])
        try:
            g.current_user = conn.execute(
                "SELECT id, username, role, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        finally:
            conn.close()

    @app.context_processor
    def inject_user() -> dict:
        """Expose the current user to templates."""
        return {"current_user": g.get("current_user")}

    # ------------------------------------------------------------------
    # Robust media download route.
    #
    # Handles BOTH absolute paths and relative filenames exactly as stored in
    # the detections table ("uploads/x.jpg", "results/detected_x.mp4",
    # "uploads/recordings/user_1_x.mp4", "screenshots/accident_x.jpg", ...).
    # Streams the real file with send_file(as_attachment=True). Every failure
    # returns a PLAIN-TEXT error (never an HTML page / redirect), so the
    # browser can never save a corrupt .htm file from this endpoint.
    # ------------------------------------------------------------------
    @app.route("/download/<path:filename>")
    def download_file(filename: str):
        # Manual auth check with a plain-text body: the login_required
        # decorator redirects to an HTML login page, which a browser could
        # otherwise save as .htm when the session has expired.
        if not session.get("user_id"):
            return _plain_text_response("Authentication required. Please log in.", 401)

        full_path = _resolve_download_file(app, filename)
        if not full_path:
            return _plain_text_response(
                f"File not found on server: {secure_filename(filename) or filename}",
                404,
            )

        mimetype = mimetypes.guess_type(full_path)[0] or "application/octet-stream"
        download_name = os.path.basename(full_path)
        try:
            return send_file(
                full_path,
                mimetype=mimetype,
                as_attachment=True,
                download_name=download_name,
                conditional=True,
            )
        except OSError as exc:
            app.logger.exception("Failed to send file %s: %s", full_path, exc)
            return _plain_text_response("Server failed to stream the requested file.", 500)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
