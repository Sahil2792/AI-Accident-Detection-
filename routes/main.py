"""Main routes and view handlers for the application."""

import json
import os
import time
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, Response, current_app, abort, send_from_directory, send_file, g
from urllib.parse import urlparse, unquote
from werkzeug.utils import secure_filename
from database.init_db import get_connection
from pathlib import Path
from utils.auth import login_required
from detector.image import process_image
from detector.annotate import save_detected_image
from detector.video import process_video as process_video_yolo
from detector.severity import classify_severity
from reports.pdf_generator import generate_pdf_report
import mimetypes
import threading
from utils.alerts import notify_accident

main_bp = Blueprint("main", __name__)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}
from detector.webcam import clear_webcam_stream, get_webcam_stream
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "flv"}


def allowed_image_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed video extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def _to_static_relative_path(file_path: str | None) -> str | None:
    """Convert an absolute or static path into a database-safe static-relative path."""
    if not file_path:
        return None

    normalized_path = str(file_path).replace("\\", "/")
    if normalized_path.startswith("/static/"):
        return normalized_path.replace("/static/", "", 1)
    if normalized_path.startswith("static/"):
        return normalized_path[len("static/"):]

    static_folder = os.path.abspath(current_app.static_folder)
    absolute_path = os.path.abspath(normalized_path)
    try:
        relative_path = os.path.relpath(absolute_path, static_folder)
    except ValueError:
        return normalized_path

    if relative_path.startswith(".."):
        return normalized_path
    return relative_path.replace("\\", "/")


def _static_url(path: str | None) -> str | None:
    """Convert a stored static-relative path into a browser URL."""
    relative_path = _to_static_relative_path(path)
    if not relative_path:
        return None
    return url_for("static", filename=relative_path)


def _media_url(path: str | None) -> str | None:
    """Convert a stored path into a browser URL, handling paths outside static/.

    Video/image uploads are stored as static-relative paths (e.g. ``uploads/x.mp4``)
    and live inside the Flask ``static/`` folder. Webcam recordings are saved under
    ``uploads/recordings/`` (project root), which is NOT inside ``static/``. This
    helper checks both locations and routes webcam files through the dedicated
    ``/media/<path>`` endpoint so the browser can play them instead of 404ing.
    """
    if not path:
        return None

    normalized_path = str(path).replace("\\", "/")

    # Already a static URL path -> serve directly from Flask static.
    if normalized_path.startswith("/static/"):
        return url_for("static", filename=normalized_path[len("/static/"):])
    if normalized_path.startswith("static/"):
        return url_for("static", filename=normalized_path[len("static/"):])

    # 1) Try the static folder first (video/image uploads are stored relative to static).
    static_folder = os.path.abspath(current_app.static_folder)
    static_candidate = os.path.abspath(os.path.join(static_folder, normalized_path))
    if os.path.exists(static_candidate):
        return url_for("static", filename=normalized_path)

    # 2) Try the project root (webcam recordings are stored relative to project root).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_candidate = os.path.abspath(os.path.join(project_root, normalized_path))
    if os.path.exists(root_candidate):
        return url_for("main.media_file", filename=normalized_path)

    # 3) Fallback: use the static URL (may 404 but at least it is a valid URL).
    return url_for("static", filename=normalized_path)


def _file_exists_on_disk(media_path) -> bool:
    """Return True when a stored media path resolves to an actual file on disk.

    Checks both the Flask ``static/`` folder (image/video uploads) and the
    project root (webcam recordings under ``uploads/recordings/``), matching
    the order used by ``_media_url``.
    """
    if not media_path:
        return False

    normalized = str(media_path).replace("\\", "/")
    static_folder = os.path.abspath(current_app.static_folder)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if normalized.startswith("/static/"):
        candidate = os.path.join(static_folder, normalized[len("/static/"):])
    elif normalized.startswith("static/"):
        candidate = os.path.join(static_folder, normalized[len("static/"):])
    elif os.path.isabs(normalized):
        candidate = normalized
    else:
        candidate = os.path.join(static_folder, normalized)
        if not os.path.exists(candidate):
            candidate = os.path.join(project_root, normalized)

    return os.path.isfile(candidate)


def _parse_json_value(raw_value):
    """Safely parse stored JSON while tolerating legacy plain-text values."""
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    try:
        parsed_value = json.loads(raw_value)
        return parsed_value if isinstance(parsed_value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _format_summary_text(summary_counts: dict, status: str, confidence: float, total_objects: int) -> str:
    """Build a clean text summary for history records and API responses."""
    lines = ["Detection Summary", "-----------------"]
    labels = (
        ("car", "Cars"),
        ("truck", "Trucks"),
        ("bus", "Buses"),
        ("motorcycle", "Motorcycles"),
        ("person", "Persons"),
        ("bicycle", "Bicycles"),
    )
    for key, label in labels:
        lines.append(f"{label}: {summary_counts.get(key, 0)}")

    lines.extend(
        [
            "",
            "Accident Status:",
            f"✅ {status}",
            "",
            "Accident Confidence:",
            f"{confidence * 100:.2f}%",
            "",
            "Total Objects Detected:",
            str(total_objects),
        ]
    )
    return "\n".join(lines)


def _download_error(message: str, status: int) -> Response:
    """Plain-text download error so browsers never save an HTML/JSON body as .htm."""
    return Response(message + "\n", status=status, mimetype="text/plain")


def _current_user_is_admin() -> bool:
    """Resilient admin check for permission-gated routes.

    Order of trust:
      1. ``g.current_user`` — freshly loaded from the DB on every request by
         the ``load_current_user`` before-request hook, so it reflects the
         user's CURRENT database role even if the session cookie is stale.
      2. Direct DB lookup by ``session['user_id']`` as a final fallback
         (covers cases where ``g.current_user`` was not populated).
    A bare ``session.get('role')`` is intentionally NOT trusted here because
    sessions created by older logins may lack/stale the role key, which caused
    valid admins to receive "Only admin may delete users."
    """
    current = g.get("current_user")
    if current:
        try:
            if str(current["role"]).strip().lower() == "admin":
                return True
        except (KeyError, IndexError, TypeError):
            pass

    uid = session.get("user_id")
    if not uid:
        return False

    try:
        conn = get_connection(current_app.config["DATABASE_PATH"])
        try:
            row = conn.execute("SELECT role FROM users WHERE id = ?", (uid,)).fetchone()
            return bool(row and str(row["role"]).strip().lower() == "admin")
        finally:
            conn.close()
    except Exception:
        current_app.logger.exception("Admin role lookup failed for user_id=%s", uid)
        return False


def _resolve_recording_path(recording_path: str) -> Path:
    """Resolve stored recording paths to an absolute filesystem path.

    Webcam recordings are stored relative to the project root
    (``uploads/recordings/user_1_xxx.mp4``); some legacy rows use ``static/`` or
    absolute paths. Resolution is anchored to the project root derived from
    this module's own location so playback/downloads keep working even when the
    Flask process is started from a different working directory.
    """
    project_root = Path(__file__).resolve().parent.parent
    normalized = str(recording_path).replace("\\", "/")

    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate

    if normalized.startswith("/static/"):
        return project_root / "static" / normalized[len("/static/"):]
    if normalized.startswith("static/"):
        return project_root / normalized

    # Prefer the project-root uploads (where webcam recordings live), then
    # fall back to the Flask static folder for static-relative rows.
    root_candidate = project_root / normalized
    static_candidate = project_root / "static" / normalized
    if root_candidate.exists():
        return root_candidate
    if static_candidate.exists():
        return static_candidate
    return root_candidate


def _persist_image_detection(
    conn,
    user_id: int,
    filename: str,
    original_image_path: str,
    detected_image_path: str,
    detection_results: dict,
    screenshot_path: str | None = None,
) -> int:
    """Store image detection results so they remain available after restart."""
    summary_counts = detection_results.get("summary", {}) or {}
    accident_status = detection_results.get("status", "Non-Accident")
    accident_confidence = float(detection_results.get("accident_confidence", 0.0))
    severity = classify_severity(accident_confidence, accident_status)
    total_objects = int(detection_results.get("total_objects", 0))
    summary_text = detection_results.get("summary_text") or _format_summary_text(
        summary_counts,
        accident_status,
        accident_confidence,
        total_objects,
    )

    payload = {
        "counts": summary_counts,
        "status": accident_status,
        "confidence": round(accident_confidence, 4),
        "severity": severity,
        "screenshot_path": screenshot_path,
        "total_objects": total_objects,
        "summary_text": summary_text,
    }

    cursor = conn.execute(
        """
        INSERT INTO detections (
            user_id, title, detection_type, media_type, severity, confidence,
            class_name, image_path, frame_path, screenshot_path, status, total_objects,
            object_counts, session_summary
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            f"Image Detection - {filename}",
            "accident" if accident_status == "Accident" else "non_accident",
            "Image",
            severity,
            accident_confidence,
            "accident" if accident_status == "Accident" else "non_accident",
            original_image_path,
            detected_image_path,
            screenshot_path,
            "completed",
            total_objects,
            json.dumps(summary_counts, ensure_ascii=False),
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    last_id = cursor.lastrowid

    # If this was an accident, notify asynchronously (best-effort)
    try:
        if str(accident_status).lower() == "accident":
            rec = {
                "id": last_id,
                "status": accident_status,
                "severity": severity,
                "confidence": accident_confidence,
                "media_type": "Image",
                "screenshot_path": screenshot_path,
                "image_path": original_image_path,
                "frame_path": detected_image_path,
                "title": f"Image Detection - {filename}",
                "created_at": None,
            }
            threading.Thread(
                target=notify_accident,
                args=(last_id, rec, current_app.static_folder, current_app.root_path),
                daemon=True,
            ).start()
    except Exception:
        # Never allow alerting to break persistence
        try:
            current_app.logger.exception("Failed to schedule image alert")
        except Exception:
            pass

    return last_id


def _persist_video_detection(
    conn,
    user_id: int,
    filename: str,
    original_video_path: str,
    output_video_path: str,
    detection_results: dict,
    screenshot_path: str | None = None,
) -> int:
    """Store video detection results so they appear in history alongside images/webcam."""
    summary_counts = detection_results.get("summary", {}) or {}
    accident_status = detection_results.get("status", "Non-Accident")
    accident_confidence = float(detection_results.get("accident_confidence", 0.0))
    severity = classify_severity(accident_confidence, accident_status)
    total_objects = int(detection_results.get("total_objects", 0))
    total_frames = int(detection_results.get("total_frames", 0))
    processed_frames = int(detection_results.get("processed_frames", 0))
    inference_time = float(detection_results.get("inference_time", 0.0))

    summary_text = _format_summary_text(
        summary_counts,
        accident_status,
        accident_confidence,
        total_objects,
    )

    payload = {
        "counts": summary_counts,
        "status": accident_status,
        "confidence": round(accident_confidence, 4),
        "severity": severity,
        "screenshot_path": screenshot_path,
        "total_objects": total_objects,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "inference_time": inference_time,
        "summary_text": summary_text,
    }

    cursor = conn.execute(
        """
        INSERT INTO detections (
            user_id, title, detection_type, media_type, severity, confidence,
            class_name, image_path, output_video_path, screenshot_path, status, total_frames, average_fps,
            total_objects, object_counts, session_summary
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            f"Video Detection - {filename}",
            "accident" if accident_status == "Accident" else "non_accident",
            "Video",
            severity,
            accident_confidence,
            "accident" if accident_status == "Accident" else "non_accident",
            original_video_path,
            output_video_path,
            screenshot_path,
            "completed",
            total_frames,
            round(total_frames / inference_time, 2) if inference_time > 0 else 0.0,
            total_objects,
            json.dumps(summary_counts, ensure_ascii=False),
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    last_id = cursor.lastrowid

    # If this was an accident, notify asynchronously (best-effort)
    try:
        if str(accident_status).lower() == "accident":
            rec = {
                "id": last_id,
                "status": accident_status,
                "severity": severity,
                "confidence": accident_confidence,
                "media_type": "Video",
                "screenshot_path": screenshot_path,
                "image_path": original_video_path,
                "output_video_path": output_video_path,
                "title": f"Video Detection - {filename}",
                "created_at": None,
            }
            threading.Thread(
                target=notify_accident,
                args=(last_id, rec, current_app.static_folder, current_app.root_path),
                daemon=True,
            ).start()
    except Exception:
        try:
            current_app.logger.exception("Failed to schedule video alert")
        except Exception:
            pass

    return last_id


@main_bp.route("/", methods=["GET", "POST"])
@main_bp.route("/login", methods=["GET", "POST"])
def login():
    """Handle login page and authentication."""
    if session.get("user_id"):
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_connection("database.db")
        user = conn.execute(
            "SELECT id, username, role FROM users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        conn.close()

        if user:
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            flash("Login successful", "success")
            return redirect(url_for("main.dashboard"))

        flash("Invalid credentials", "danger")

    return render_template("login.html")


@main_bp.route("/logout")
def logout():
    """Clear current session and redirect to login."""
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("main.login"))


@main_bp.route("/register", methods=["GET", "POST"])
def register():
    """Render the registration page and create a new user."""
    if session.get("user_id"):
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        # FIX: Agar user email nahi deta, toh auto-generated fallback email set karein
        email_input = request.form.get("email", "").strip()
        if email_input:
            email = email_input
        else:
            email = f"{username.lower().replace(' ', '')}@example.com"

        if not username or not password:
            flash("Username and password are required", "warning")
            return render_template("register.html")

        conn = get_connection(current_app.config["DATABASE_PATH"])
        try:
            existing_user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if existing_user:
                flash("Username already exists", "danger")
                return render_template("register.html")

            conn.execute(
                "INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, 'user')",
                (username, password, email),
            )
            conn.commit()
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("main.login"))
        finally:
            conn.close()

    return render_template("register.html")


@main_bp.route("/dashboard")
@login_required
def dashboard():
    """Render the main dashboard page."""
    conn = get_connection("database.db")
    user_id = session.get("user_id")

    # Filter by user_id for all stats to ensure dashboard reflects only user's data
    if session.get("role") == "admin":
        # Admin sees all detections
        total_detections = conn.execute("SELECT COUNT(*) AS count FROM detections").fetchone()["count"]
        accidents = conn.execute("SELECT COUNT(*) AS count FROM detections WHERE detection_type = 'accident'").fetchone()["count"]
        non_accidents = conn.execute("SELECT COUNT(*) AS count FROM detections WHERE detection_type = 'non_accident'").fetchone()["count"]
        high_severity = conn.execute("SELECT COUNT(*) AS count FROM detections WHERE severity = 'high'").fetchone()["count"]
    else:
        # Regular users see only their own detections
        total_detections = conn.execute("SELECT COUNT(*) AS count FROM detections WHERE user_id = ?", (user_id,)).fetchone()["count"]
        accidents = conn.execute("SELECT COUNT(*) AS count FROM detections WHERE user_id = ? AND detection_type = 'accident'", (user_id,)).fetchone()["count"]
        non_accidents = conn.execute("SELECT COUNT(*) AS count FROM detections WHERE user_id = ? AND detection_type = 'non_accident'", (user_id,)).fetchone()["count"]
        high_severity = conn.execute("SELECT COUNT(*) AS count FROM detections WHERE user_id = ? AND severity = 'high'", (user_id,)).fetchone()["count"]

    conn.close()

    return render_template(
        "dashboard.html",
        total_detections=total_detections,
        accidents=accidents,
        non_accidents=non_accidents,
        high_severity=high_severity,
    )


@main_bp.route("/image")
@login_required
def image_detection():
    """Render the image detection placeholder page."""
    return render_template("image.html")


@main_bp.route("/api/detect-image", methods=["POST"])
@login_required
def detect_image_api():
    """Handle image upload and return YOLO detection results."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_image_file(file.filename):
        return jsonify({"error": "Invalid file format. Allowed: jpg, jpeg, png, bmp, gif, webp"}), 400

    try:
        upload_folder = current_app.config["UPLOAD_FOLDER"]
        result_folder = current_app.config["RESULT_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(result_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        upload_path = os.path.join(upload_folder, unique_filename)
        file.save(upload_path)

        start_time = time.time()
        detection_results = process_image(
            upload_path,
            model_path=current_app.config.get("YOLO_MODEL_PATH"),
            accident_confidence_threshold=current_app.config.get("ACCIDENT_CONFIDENCE_THRESHOLD"),
        )
        inference_time = round(time.time() - start_time, 2)

        annotated_filename = f"annotated_{timestamp}_{filename}"
        annotated_path = os.path.join(result_folder, annotated_filename)
        save_detected_image(upload_path, detection_results.get("objects", []), annotated_path)

        conn = get_connection(current_app.config["DATABASE_PATH"])
        try:
            screenshot_rel = (
                _to_static_relative_path(annotated_path)
                if (detection_results.get("status") == "Accident")
                else None
            )
            det_id = _persist_image_detection(
                conn,
                session["user_id"],
                filename,
                _to_static_relative_path(upload_path),
                _to_static_relative_path(annotated_path),
                detection_results,
                screenshot_path=screenshot_rel,
            )
            conn.commit()
        finally:
            conn.close()

        summary_counts = detection_results.get("summary", {}) or {}
        summary_text = detection_results.get("summary_text") or _format_summary_text(
            summary_counts,
            detection_results.get("status", "Non-Accident"),
            float(detection_results.get("accident_confidence", 0.0)),
            int(detection_results.get("total_objects", 0)),
        )

        return jsonify(
            {
                "success": True,
                "original_image": url_for("static", filename=f"uploads/{unique_filename}"),
                "detected_image": url_for("static", filename=f"results/{annotated_filename}"),
                "objects": detection_results["objects"],
                "summary": summary_counts,
                "summary_display": detection_results.get("summary_display", []),
                "summary_text": summary_text,
                "accident_status": detection_results.get("status", "Non-Accident"),
                "accident_confidence": detection_results.get("accident_confidence", 0.0),
                "severity": detection_results.get("severity", "low"),
                "screenshot": (
                    url_for("static", filename=f"results/{annotated_filename}")
                    if (detection_results.get("status") == "Accident")
                    else None
                ),
                "total_objects": detection_results["total_objects"],
                "inference_time": inference_time,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


@main_bp.route("/api/detect-video", methods=["POST"])
@login_required
def detect_video_api():
    """Handle video upload, run dual-model detection, and persist to history."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_video_file(file.filename):
        return jsonify({"error": "Invalid file format. Allowed: mp4, avi, mov, mkv, flv"}), 400

    try:
        upload_folder = current_app.config["UPLOAD_FOLDER"]
        result_folder = current_app.config["RESULT_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(result_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        upload_path = os.path.join(upload_folder, unique_filename)
        file.save(upload_path)

        output_filename = f"detected_{timestamp}_{filename}"
        output_path = os.path.join(result_folder, output_filename)

        detection_results = process_video_yolo(upload_path, output_path)

        # Persist video detection results to the database so they appear in history.
        conn = get_connection(current_app.config["DATABASE_PATH"])
        try:
            screenshot_path = detection_results.get("screenshot_path")
            screenshot_rel = (
                _to_static_relative_path(screenshot_path) if screenshot_path else None
            )
            det_id = _persist_video_detection(
                conn,
                session["user_id"],
                filename,
                _to_static_relative_path(upload_path),
                _to_static_relative_path(output_path),
                detection_results,
                screenshot_path=screenshot_rel,
            )
            conn.commit()
        finally:
            conn.close()


        return jsonify(
            {
                "success": True,
                "original_video": f"/static/uploads/{unique_filename}",
                "detected_video": f"/static/results/{output_filename}",
                "total_objects": detection_results["total_objects"],
                "total_frames": detection_results["total_frames"],
                "processed_frames": detection_results["processed_frames"],
                "summary": detection_results.get("summary", {}),
                "average_confidence": detection_results["average_confidence"],
                "accident_confidence": detection_results.get("accident_confidence", 0.0),
                "accident_incidents": detection_results.get("accident_incidents", 0),
                "status": detection_results.get("status", "Non-Accident"),
                "severity": detection_results.get("severity", "low"),
                "screenshot": (_media_url(screenshot_rel) if screenshot_rel else None),
                "inference_time": detection_results["inference_time"],
            }
        )

    except Exception as e:
        return jsonify({"error": f"Video detection failed: {str(e)}"}), 500


@main_bp.route("/video")
@login_required
def video_detection():
    """Render the video detection placeholder page."""
    return render_template("video.html")


@main_bp.route("/webcam")
@login_required
def webcam_detection():
    """Render the live webcam page with the latest saved recording."""
    conn = get_connection(current_app.config["DATABASE_PATH"])
    try:
        latest = conn.execute(
            """
            SELECT * FROM detections
            WHERE media_type = 'Webcam' AND user_id = ?
            ORDER BY created_at DESC, id DESC LIMIT 1
            """,
            (session["user_id"],),
        ).fetchone()
    finally:
        conn.close()

    latest_recording = None
    if latest:
        latest_recording = {
            "recording_name": latest["title"],
            "review_url": url_for("main.recording_review", detection_id=latest["id"]),
            "download_url": url_for("main.download_recording", detection_id=latest["id"]),
        }

    return render_template("webcam.html", latest_recording=latest_recording)


@main_bp.route("/webcam_feed")
@login_required
def webcam_feed():
    """Start the webcam stream and return the MJPEG response.

    Accepts an optional ``camera_source`` (or ``source``) query parameter so
    users can select a camera index (e.g. ``0`` for DroidCam) or an IP stream
    URL (e.g. ``http://<ip>:<port>/video``).
    """
    user_id = session.get("user_id") or 0
    camera_source = request.args.get("camera_source") or request.args.get("source") or current_app.config.get("CAMERA_SOURCE")

    try:
        stream = get_webcam_stream(
            user_id,
            current_app.config.get("YOLO_MODEL_PATH"),
            camera_source,
        )
    except TypeError:
        try:
            stream = get_webcam_stream(user_id)
        except TypeError:
            stream = get_webcam_stream()
    stream.start()

    def _frame_generator():
        while True:
            frame = stream.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    return Response(_frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@main_bp.route("/webcam_stats")
@login_required
def webcam_stats():
    """Return live webcam detection statistics as JSON."""
    user_id = session.get("user_id") or 0
    camera_source = request.args.get("camera_source") or request.args.get("source") or current_app.config.get("CAMERA_SOURCE")
    try:
        stream = get_webcam_stream(
            user_id,
            current_app.config.get("YOLO_MODEL_PATH"),
            camera_source,
        )
        stats = stream.get_stats()
        return jsonify({"success": True, **stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@main_bp.route("/stop_webcam")
@login_required
def stop_webcam():
    """Stop the webcam stream and persist the final session once."""
    camera_source = request.args.get("camera_source") or request.args.get("source") or current_app.config.get("CAMERA_SOURCE")
    stream = get_webcam_stream(
        session["user_id"],
        model_path=current_app.config.get("YOLO_MODEL_PATH"),
        camera_source=camera_source,
    )
    summary = stream.stop()

    output_video_path = summary.get("output_video_path")
    if not output_video_path:
        return jsonify({"success": True, "summary": summary})

    conn = get_connection(current_app.config["DATABASE_PATH"])
    try:
        existing_detection = conn.execute(
            "SELECT id FROM detections WHERE output_video_path = ?",
            (output_video_path,),
        ).fetchone()
        if existing_detection:
            return jsonify({"success": True, "summary": summary, "detection_id": existing_detection[0]})

        detection_type = "accident" if summary.get("status") == "Accident" else "non_accident"
        severity = classify_severity(
            float(summary.get("accident_confidence", summary.get("confidence_score", 0.0))),
            summary.get("status"),
        )
        screenshot_path = summary.get("screenshot_path")
        screenshot_rel = _to_static_relative_path(screenshot_path) if screenshot_path else None
        title = f"Webcam Recording - {Path(output_video_path).name}"
        conn.execute(
            """
            INSERT INTO detections (
                user_id, title, detection_type, media_type, severity, confidence,
                class_name, image_path, output_video_path, screenshot_path, status, total_frames, average_fps,
                total_objects, object_counts, session_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                title,
                detection_type,
                "Webcam",
                severity,
                float(summary.get("accident_confidence", summary.get("confidence_score", 0.0))),
                detection_type,
                output_video_path,
                output_video_path,
                screenshot_rel,
                "completed",
                int(summary.get("total_frames", 0)),
                float(summary.get("average_fps", 0.0)),
                int(summary.get("total_objects_detected", summary.get("total_objects", 0))),
                json.dumps(summary.get("object_counts_by_class", {}), ensure_ascii=False),
                json.dumps(summary, ensure_ascii=False),
            ),
        )
        detection_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()[0]
        conn.execute(
            """
            INSERT INTO recordings (
                user_id, detection_id, recording_name, video_path, start_time, end_time,
                duration, detected_objects, confidence_score, accident_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                detection_id,
                Path(output_video_path).name,
                output_video_path,
                summary.get("start_time"),
                summary.get("end_time"),
                float(summary.get("duration", 0.0)),
                json.dumps(summary.get("object_counts_by_class", {}), ensure_ascii=False),
                float(summary.get("accident_confidence", summary.get("confidence_score", 0.0))),
                summary.get("status", "non_accident").lower().replace("-", "_"),
            ),
        )
        conn.commit()
        stream.mark_session_saved(detection_id)
    finally:
        conn.close()

    return jsonify(
        {
            "success": True,
            "summary": summary,
            "severity": severity,
            "screenshot_url": _media_url(screenshot_rel) if screenshot_rel else None,
            "detection_id": summary.get("detection_id") or detection_id,
        }
    )


@main_bp.route("/media/<path:filename>")
@login_required
def media_file(filename):
    """Serve files from the project's uploads directory (webcam recordings).

    Webcam recordings are saved under ``uploads/recordings/`` which is outside
    the Flask ``static/`` folder. This endpoint streams those MP4 files so the
    history page's "View Original" / "View Detected" buttons can play them.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_folder = os.path.join(project_root, "static")
    normalized = str(filename).replace("\\", "/").lstrip("/")

    # Resolve relative to the project root AND the static folder so paths like
    # ``uploads/recordings/user_1_20240101.mp4`` (project-root uploads) or
    # legacy ``recordings/webcam/...`` map to the correct file on disk.
    candidates = [
        os.path.abspath(os.path.join(project_root, normalized)),
        os.path.abspath(os.path.join(static_folder, normalized)),
    ]
    full_path = next((p for p in candidates if os.path.isfile(p)), None)
    if not full_path:
        abort(404)

    # Prevent serving anything outside the media directories.
    safe_bases = [
        os.path.abspath(os.path.join(project_root, "uploads")),
        os.path.abspath(os.path.join(project_root, "recordings")),
        os.path.abspath(static_folder),
    ]
    for base in safe_bases:
        try:
            if os.path.commonpath([full_path, base]) == base:
                return send_from_directory(
                    os.path.dirname(full_path),
                    os.path.basename(full_path),
                )
        except ValueError:
            continue

    abort(404)


@main_bp.route("/recording/<int:detection_id>")
@login_required
def recording_review(detection_id):
    """Render the webcam recording review page."""
    conn = get_connection(current_app.config["DATABASE_PATH"])
    detection = conn.execute(
        "SELECT * FROM detections WHERE id = ?",
        (detection_id,),
    ).fetchone()
    conn.close()

    if not detection:
        flash("Recording not found", "warning")
        return redirect(url_for("main.history"))

    if session.get("role") != "admin" and detection["user_id"] != session.get("user_id"):
        flash("Access denied", "danger")
        return redirect(url_for("main.history"))

    return render_template("recording_review.html", detection=detection)


@main_bp.route("/recording/<int:detection_id>/video")
@login_required
def recording_video_feed(detection_id):
    """Stream a webcam recording video for inline playback."""
    conn = get_connection(current_app.config["DATABASE_PATH"])
    detection = conn.execute(
        "SELECT * FROM detections WHERE id = ?",
        (detection_id,),
    ).fetchone()
    conn.close()

    if not detection:
        abort(404)

    if session.get("role") != "admin" and detection["user_id"] != session.get("user_id"):
        abort(403)

    video_path = detection["output_video_path"]
    if not video_path:
        abort(404)

    absolute_path = _resolve_recording_path(video_path)
    if not os.path.exists(absolute_path):
        abort(404)

    return send_from_directory(
        os.path.dirname(absolute_path),
        os.path.basename(absolute_path),
        mimetype="video/mp4",
    )


@main_bp.route("/recording/<int:detection_id>/download")
@login_required
def download_recording(detection_id):
    """Download a webcam recording as an MP4 attachment.

    Returns JSON errors (404/403/500) to avoid HTML error pages being downloaded
    by clients in error cases.
    """
    conn = get_connection(current_app.config["DATABASE_PATH"])
    detection = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    conn.close()

    if not detection:
        return jsonify({"error": "detection_not_found", "message": "Recording not found."}), 404

    if session.get("role") != "admin" and detection["user_id"] != session.get("user_id"):
        return jsonify({"error": "forbidden", "message": "Access denied."}), 403

    video_path = detection["output_video_path"]
    if not video_path:
        return jsonify({"error": "file_not_recorded", "message": "No recorded video path present."}), 404

    absolute_path = _resolve_recording_path(video_path)
    if not absolute_path or not os.path.exists(absolute_path):
        current_app.logger.warning("Recording download requested but file missing: %s (resolved: %s)", video_path, absolute_path)
        return jsonify({"error": "file_not_found", "message": "Recording file not found on server."}), 404

    try:
        return send_file(absolute_path, as_attachment=True, download_name=os.path.basename(absolute_path), mimetype="video/mp4")
    except Exception as e:
        current_app.logger.exception("Failed to send recording %s: %s", absolute_path, e)
        return jsonify({"error": "send_failed", "message": "Server failed to send recording."}), 500


@main_bp.route("/history")
@login_required
def history():
    """Render the detection history page with optional filtering.

    Query parameters:
        - ``filter``: ``all``, ``accident``, or ``non_accident``
        - ``media``: ``all``, ``image``, ``video``, or ``webcam``
        - ``q``: free-text search on title
    """
    filter_type = request.args.get("filter", "all")
    media_type = request.args.get("media", "all")
    search_query = request.args.get("q", "").strip().lower()

    conn = get_connection(current_app.config["DATABASE_PATH"])
    query = "SELECT * FROM detections"
    conditions = []
    params = []

    if session.get("role") != "admin":
        conditions.append("user_id = ?")
        params.append(session.get("user_id"))

    if filter_type == "accident":
        conditions.append("detection_type = 'accident'")
    elif filter_type == "non_accident":
        conditions.append("detection_type = 'non_accident'")

    if media_type == "image":
        conditions.append("media_type = 'Image'")
    elif media_type == "video":
        conditions.append("media_type = 'Video'")
    elif media_type == "webcam":
        conditions.append("media_type = 'Webcam'")

    if search_query:
        conditions.append("LOWER(title) LIKE ?")
        params.append(f"%{search_query}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY created_at DESC, id DESC LIMIT 100"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    detections = []
    for row_index, row in enumerate(rows, start=1):
        record = dict(row)
        record["row_number"] = row_index
        summary_counts = _parse_json_value(record.get("object_counts"))
        session_summary = _parse_json_value(record.get("session_summary"))
        if session_summary.get("counts"):
            summary_counts = session_summary["counts"]

        confidence_value = session_summary.get("confidence", record.get("confidence", 0.0))
        if confidence_value is None:
            confidence_value = 0.0

        media_type_lower = str(record.get("media_type") or "").lower()
        if media_type_lower == "image":
            # Images: image_path = original, frame_path = detected/annotated
            original_path = record.get("image_path")
            detected_path = record.get("frame_path")
        elif media_type_lower == "video":
            # Videos: image_path stores the ORIGINAL uploaded video,
            # output_video_path stores the DETECTED (processed) video.
            original_path = record.get("image_path") or record.get("output_video_path")
            detected_path = record.get("output_video_path")
        else:
            # Webcam: only the recording exists (output_video_path)
            original_path = record.get("output_video_path")
            detected_path = record.get("output_video_path")

        record.update(
            {
                "summary_counts": summary_counts,
                "summary_text": session_summary.get("summary_text") or record.get("session_summary") or "",
                "accident_confidence_percent": f"{float(confidence_value) * 100:.2f}%",
                "accident_status": session_summary.get("status") or record.get("detection_type") or record.get("status") or "",
                "original_image_url": _media_url(original_path) if _file_exists_on_disk(original_path) else None,
                "detected_image_url": _media_url(detected_path) if _file_exists_on_disk(detected_path) else None,
                # Raw stored paths (not URLs) for the robust download endpoint.
                # Gated on existence so missing files never render dead links that
                # would save a .txt 404 body or open a black video player.
                "original_download_path": original_path if _file_exists_on_disk(original_path) else None,
                "detected_download_path": detected_path if _file_exists_on_disk(detected_path) else None,
                "screenshot_url": _media_url(record.get("screenshot_path")) if _file_exists_on_disk(record.get("screenshot_path")) else None,
                "recorded_at": record.get("created_at"),
            }
        )
        detections.append(record)

    return render_template(
        "history.html",
        detections=detections,
        current_filter=filter_type,
        current_media=media_type,
        search_query=request.args.get("q", ""),
    )


@main_bp.route("/history/delete_all", methods=["POST"])
@login_required
def delete_all_detections():
    """Delete all detection history records for the current user (or all for admin)."""
    conn = get_connection(current_app.config["DATABASE_PATH"])
    try:
        if session.get("role") == "admin":
            conn.execute("DELETE FROM recordings")
            conn.execute("DELETE FROM detections")
        else:
            conn.execute("DELETE FROM recordings WHERE user_id = ?", (session.get("user_id"),))
            conn.execute("DELETE FROM detections WHERE user_id = ?", (session.get("user_id"),))
        conn.commit()
        flash("All detection records deleted successfully", "success")
    except Exception as e:
        flash(f"Failed to delete records: {str(e)}", "danger")
    finally:
        conn.close()

    return redirect(url_for("main.history"))


@main_bp.route("/history/<int:detection_id>/delete", methods=["POST"])
@login_required
def delete_detection(detection_id):
    """Delete a detection history record."""
    conn = get_connection(current_app.config["DATABASE_PATH"])
    detection = conn.execute(
        "SELECT * FROM detections WHERE id = ?",
        (detection_id,),
    ).fetchone()

    if not detection:
        recording = conn.execute(
            "SELECT * FROM recordings WHERE id = ?",
            (detection_id,),
        ).fetchone()
        if not recording:
            conn.close()
            flash("Detection record not found", "warning")
            return redirect(url_for("main.history"))

        conn.execute("DELETE FROM recordings WHERE id = ?", (detection_id,))
        if recording["detection_id"]:
            conn.execute("DELETE FROM detections WHERE id = ?", (recording["detection_id"],))
        conn.commit()
        conn.close()

        flash("Detection deleted successfully", "success")
        return redirect(url_for("main.history"))

    # Check permission: non-admin users can only delete their own records
    if session.get("role") != "admin" and detection["user_id"] != session.get("user_id"):
        conn.close()
        flash("Access denied", "danger")
        return redirect(url_for("main.history"))

    conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    conn.execute("DELETE FROM recordings WHERE detection_id = ?", (detection_id,))
    conn.commit()
    conn.close()

    flash("Detection deleted successfully", "success")
    return redirect(url_for("main.history"))


@main_bp.route("/download/<int:detection_id>/<file_type>")
@login_required
def download_file(detection_id, file_type):
    """Download original or processed file for a detection record.

    This route returns a binary attachment on success. On failure it returns a
    JSON error response with an appropriate HTTP status code (avoids returning
    an HTML error page which browsers may save as .htm).
    """
    conn = get_connection(current_app.config["DATABASE_PATH"])
    detection = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    conn.close()

    if not detection:
        return _download_error("Detection record not found.", 404)

    # Permission check
    if session.get("role") != "admin" and detection["user_id"] != session.get("user_id"):
        return _download_error("Access denied.", 403)

    # Choose column depending on requested file type
    if file_type == "original":
        file_path = detection["image_path"] or detection["output_video_path"]
        if not file_path:
            return _download_error("Original file not recorded for this detection.", 404)
    elif file_type == "processed":
        file_path = detection["frame_path"] or detection["output_video_path"]
        if not file_path:
            return _download_error("Processed file not recorded for this detection.", 404)
    else:
        return _download_error("file_type must be 'original' or 'processed'.", 400)

    raw = str(file_path).strip()
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https"):
        path = unquote(parsed.path)
    else:
        path = unquote(raw)

    absolute_path = None

    # Try explicit /static/ or static/ prefixes first
    if path.startswith("/static/"):
        rel = path[len("/static/") :]
        candidate = os.path.join(current_app.static_folder, rel)
        if os.path.exists(candidate):
            absolute_path = os.path.abspath(candidate)
    elif path.startswith("static/"):
        rel = path[len("static/") :]
        candidate = os.path.join(current_app.static_folder, rel)
        if os.path.exists(candidate):
            absolute_path = os.path.abspath(candidate)
    else:
        # Absolute path stored directly
        if os.path.isabs(path) and os.path.exists(path):
            absolute_path = os.path.abspath(path)
        else:
            # Try common project-relative locations in order
            # 1) configured UPLOAD_FOLDER (often static/uploads)
            upload_folder = current_app.config.get("UPLOAD_FOLDER") or os.path.join(current_app.static_folder, "uploads")
            candidate = os.path.abspath(os.path.join(upload_folder, path))
            if os.path.exists(candidate):
                absolute_path = candidate
            else:
                # 2) static folder fallback
                candidate2 = os.path.abspath(os.path.join(current_app.static_folder, path))
                if os.path.exists(candidate2):
                    absolute_path = candidate2
                else:
                    # 3) project root relative (e.g., uploads/recordings/...)
                    candidate3 = os.path.abspath(os.path.join(current_app.root_path, path))
                    if os.path.exists(candidate3):
                        absolute_path = candidate3
                    else:
                        # 4) raw relative path resolved against project root
                        candidate4 = os.path.abspath(path)
                        if os.path.exists(candidate4):
                            absolute_path = candidate4

    # Final check
    if not absolute_path or not os.path.exists(absolute_path):
        # Plain-text 404 (never HTML/redirect) so the browser cannot save a
        # corrupt .htm file when the media is missing on disk.
        current_app.logger.warning("Download requested but file missing: %s (resolved: %s)", raw, absolute_path)
        return _download_error("File not found on server.", 404)

    # Security: only allow downloads from safe directories (static and uploads inside project)
    allowed_dirs = [os.path.abspath(current_app.static_folder), os.path.abspath(os.path.join(current_app.root_path, "uploads"))]
    abs_allowed = False
    for ad in allowed_dirs:
        try:
            # Using commonpath to ensure absolute_path is inside allowed dir
            if os.path.commonpath([absolute_path, ad]) == ad:
                abs_allowed = True
                break
        except ValueError:
            continue

    if not abs_allowed:
        current_app.logger.warning("Attempt to download file outside allowed directories: %s", absolute_path)
        return _download_error("Requested file is not available for download.", 403)

    # Build a safe download filename
    actual_filename = os.path.basename(absolute_path)
    download_filename = f"{file_type}_{actual_filename}"

    # Guess mimetype for correct browser handling
    import mimetypes

    mime_type, _ = mimetypes.guess_type(absolute_path)
    try:
        return send_file(absolute_path, as_attachment=True, download_name=download_filename, mimetype=mime_type)
    except Exception as e:
        current_app.logger.exception("Failed to send file %s: %s", absolute_path, e)
        return _download_error("Server failed to send file.", 500)


@main_bp.route('/reports/generate', methods=['POST'])
@login_required
def generate_report_post():
    """Generate a PDF report from the selected detection (form POST) and stream it to user."""
    detection_id = request.form.get('detection_id')
    if not detection_id:
        return jsonify({"error": "missing_parameter", "message": "detection_id is required."}), 400
    try:
        detection_id = int(detection_id)
    except ValueError:
        return jsonify({"error": "invalid_parameter", "message": "detection_id must be an integer."}), 400

    # Reuse existing generator logic by fetching the detection and calling generate_pdf_report
    conn = get_connection(current_app.config["DATABASE_PATH"])
    detection_row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    conn.close()

    if not detection_row:
        return jsonify({"error": "detection_not_found", "message": "Detection record not found."}), 404

    detection = dict(detection_row)
    if session.get("role") != "admin" and detection.get("user_id") != session.get("user_id"):
        return jsonify({"error": "forbidden", "message": "Access denied."}), 403

    session_summary = None
    try:
        if detection.get("session_summary"):
            session_summary = json.loads(detection.get("session_summary"))
    except Exception:
        session_summary = None

    payload = {
        "id": detection.get("id"),
        "title": detection.get("title") or f"Detection {detection.get('id')}",
        "media_type": detection.get("media_type"),
        "detection_type": detection.get("detection_type"),
        "status": session_summary.get("status") if session_summary and session_summary.get("status") else detection.get("status") or detection.get("detection_type"),
        "confidence": session_summary.get("confidence") if session_summary and session_summary.get("confidence") is not None else detection.get("confidence"),
        "severity": detection.get("severity"),
        "created_at": detection.get("created_at"),
        "object_counts": detection.get("object_counts"),
        "session_summary": detection.get("session_summary"),
        "screenshot_path": detection.get("screenshot_path") or detection.get("frame_path") or detection.get("image_path"),
    }
    payload['_app'] = current_app

    try:
        pdf_path = generate_pdf_report(payload)
    except Exception as e:
        current_app.logger.exception("Failed to generate PDF for detection %s: %s", detection_id, e)
        return jsonify({"error": "generate_failed", "message": "Failed to generate PDF."}), 500

    if not os.path.exists(pdf_path):
        return jsonify({"error": "pdf_missing", "message": "Generated PDF missing."}), 500

    try:
        mime_type, _ = mimetypes.guess_type(pdf_path)
        return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path), mimetype=mime_type or 'application/pdf')
    except Exception as e:
        current_app.logger.exception("Failed to send PDF %s: %s", pdf_path, e)
        return jsonify({"error": "send_failed", "message": "Failed to send PDF."}), 500


@main_bp.route('/reports/generate/<int:detection_id>')
@login_required
def generate_report(detection_id):
    """Backward-compatible single-detection generator that streams the PDF (GET)."""
    # delegate to the POST implementation for consistent behavior
    # Build a fake request.form by calling the POST function logic directly
    # For simplicity, just fetch and generate as in the POST handler
    conn = get_connection(current_app.config["DATABASE_PATH"])
    detection_row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    conn.close()

    if not detection_row:
        return jsonify({"error": "detection_not_found", "message": "Detection record not found."}), 404

    detection = dict(detection_row)
    if session.get("role") != "admin" and detection.get("user_id") != session.get("user_id"):
        return jsonify({"error": "forbidden", "message": "Access denied."}), 403

    session_summary = None
    try:
        if detection.get("session_summary"):
            session_summary = json.loads(detection.get("session_summary"))
    except Exception:
        session_summary = None

    payload = {
        "id": detection.get("id"),
        "title": detection.get("title") or f"Detection {detection.get('id')}",
        "media_type": detection.get("media_type"),
        "detection_type": detection.get("detection_type"),
        "status": session_summary.get("status") if session_summary and session_summary.get("status") else detection.get("status") or detection.get("detection_type"),
        "confidence": session_summary.get("confidence") if session_summary and session_summary.get("confidence") is not None else detection.get("confidence"),
        "severity": detection.get("severity"),
        "created_at": detection.get("created_at"),
        "object_counts": detection.get("object_counts"),
        "session_summary": detection.get("session_summary"),
        "screenshot_path": detection.get("screenshot_path") or detection.get("frame_path") or detection.get("image_path"),
    }
    payload['_app'] = current_app

    try:
        pdf_path = generate_pdf_report(payload)
    except Exception as e:
        current_app.logger.exception("Failed to generate PDF for detection %s: %s", detection_id, e)
        return jsonify({"error": "generate_failed", "message": "Failed to generate PDF."}), 500

    if not os.path.exists(pdf_path):
        return jsonify({"error": "pdf_missing", "message": "Generated PDF missing."}), 500

    try:
        mime_type, _ = mimetypes.guess_type(pdf_path)
        return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path), mimetype=mime_type or 'application/pdf')
    except Exception as e:
        current_app.logger.exception("Failed to send PDF %s: %s", pdf_path, e)
        return jsonify({"error": "send_failed", "message": "Failed to send PDF."}), 500


@main_bp.route("/analytics")
@login_required
def analytics():
    """Render analytics using the detection history database as the source of truth.

    Provides totals, severity distribution, source distribution, trends, and average confidence.
    """
    conn = get_connection(current_app.config["DATABASE_PATH"])
    try:
        params = []
        user_filter = ""
        if session.get("role") != "admin":
            user_filter = "WHERE user_id = ?"
            params.append(session.get("user_id"))

        # Totals
        total_q = "SELECT COUNT(*) AS cnt FROM detections " + user_filter
        total = conn.execute(total_q, params).fetchone()["cnt"] if params else conn.execute(total_q).fetchone()["cnt"]

        # Accidents and non-accidents
        acc_q = "SELECT detection_type, COUNT(*) AS cnt FROM detections " + user_filter + " GROUP BY detection_type"
        acc_rows = conn.execute(acc_q, params).fetchall() if params else conn.execute(acc_q).fetchall()
        accidents = 0
        non_accidents = 0
        for r in acc_rows:
            if r["detection_type"] == "accident":
                accidents = r["cnt"]
            else:
                non_accidents = r["cnt"]

        # Severity distribution
        sev_q = "SELECT severity, COUNT(*) AS cnt FROM detections " + user_filter + " GROUP BY severity"
        sev_rows = conn.execute(sev_q, params).fetchall() if params else conn.execute(sev_q).fetchall()
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        for r in sev_rows:
            key = (r["severity"] or "").lower()
            if key in severity_counts:
                severity_counts[key] = r["cnt"]

        # Source (media_type) distribution
        src_q = "SELECT media_type, COUNT(*) AS cnt FROM detections " + user_filter + " GROUP BY media_type"
        src_rows = conn.execute(src_q, params).fetchall() if params else conn.execute(src_q).fetchall()
        source_counts = {"Image": 0, "Video": 0, "Webcam": 0}
        for r in src_rows:
            mt = r["media_type"] or ""
            if mt in source_counts:
                source_counts[mt] = r["cnt"]
            else:
                source_counts[mt] = r["cnt"]

        # Average accident confidence (only accidents)
        avg_q = "SELECT AVG(confidence) AS avg_conf FROM detections " + ("WHERE detection_type = 'accident'" + (" AND user_id = ?" if session.get("role") != "admin" else ""))
        avg_params = params.copy() if params else []
        avg_row = conn.execute(avg_q, avg_params).fetchone()
        avg_confidence = float(avg_row["avg_conf"] or 0.0)

        # Daily trend (last 30 days) for accidents
        daily_labels = []
        daily_values = []
        from datetime import datetime, timedelta

        today = datetime.utcnow().date()
        days = 30
        # Build map from date->count
        start_date = today - timedelta(days=days - 1)
        date_counts = { (start_date + timedelta(days=i)).isoformat(): 0 for i in range(days) }

        daily_q = "SELECT DATE(created_at) AS d, COUNT(*) AS cnt FROM detections WHERE detection_type = 'accident'"
        daily_params = []
        if session.get("role") != "admin":
            daily_q += " AND user_id = ?"
            daily_params.append(session.get("user_id"))
        daily_q += " AND DATE(created_at) >= DATE('now','-29 day') GROUP BY d ORDER BY d"
        rows = conn.execute(daily_q, daily_params).fetchall()
        for r in rows:
            if r["d"]:
                date_counts[r["d"]] = r["cnt"]
        daily_labels = list(date_counts.keys())
        daily_values = list(date_counts.values())

        # Weekly trend (last 12 weeks) using year-week (YYYY-WW)
        weekly_labels = []
        weekly_values = []
        weekly_q = "SELECT STRFTIME('%Y-%W', created_at) AS week, COUNT(*) AS cnt FROM detections WHERE detection_type = 'accident'"
        weekly_params = []
        if session.get("role") != "admin":
            weekly_q += " AND user_id = ?"
            weekly_params.append(session.get("user_id"))
        weekly_q += " GROUP BY week ORDER BY week DESC LIMIT 12"
        wrows = conn.execute(weekly_q, weekly_params).fetchall()
        # wrows ordered desc; reverse to ascending
        wrows = list(reversed(wrows))
        for r in wrows:
            weekly_labels.append(r["week"])
            weekly_values.append(r["cnt"])

        # Monthly trend (last 12 months)
        monthly_labels = []
        monthly_values = []
        monthly_q = "SELECT STRFTIME('%Y-%m', created_at) AS month, COUNT(*) AS cnt FROM detections WHERE detection_type = 'accident'"
        monthly_params = []
        if session.get("role") != "admin":
            monthly_q += " AND user_id = ?"
            monthly_params.append(session.get("user_id"))
        monthly_q += " GROUP BY month ORDER BY month DESC LIMIT 12"
        mrows = conn.execute(monthly_q, monthly_params).fetchall()
        mrows = list(reversed(mrows))
        for r in mrows:
            monthly_labels.append(r["month"])
            monthly_values.append(r["cnt"])

    finally:
        conn.close()

    return render_template(
        "analytics.html",
        total_detections=total,
        accidents=accidents,
        non_accidents=non_accidents,
        severity_counts=severity_counts,
        source_counts=source_counts,
        avg_confidence=avg_confidence,
        daily_labels=daily_labels,
        daily_values=daily_values,
        weekly_labels=weekly_labels,
        weekly_values=weekly_values,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
    )


@main_bp.route("/reports")
@login_required
def reports():
    """Render the reports page showing generated PDF files and a small form to generate new reports.

    Falls back to the reports/ directory in the project root where generated PDFs are stored.
    """
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)

    # list files in reports directory
    files = []
    for name in sorted(os.listdir(reports_dir), reverse=True):
        if not name.lower().endswith('.pdf'):
            continue
        full = os.path.join(reports_dir, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0
        files.append({"filename": name, "path": full, "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})

    # Provide a list of recent detections so the user can choose one to generate a report for
    conn = get_connection(current_app.config["DATABASE_PATH"])
    try:
        if session.get("role") == "admin":
            recent = conn.execute("SELECT id, title, created_at FROM detections ORDER BY created_at DESC LIMIT 100").fetchall()
        else:
            recent = conn.execute("SELECT id, title, created_at FROM detections WHERE user_id = ? ORDER BY created_at DESC LIMIT 100", (session.get("user_id"),)).fetchall()
        recent_list = [dict(r) for r in recent]
    finally:
        conn.close()

    return render_template("reports.html", reports=files, recent_detections=recent_list)


@main_bp.route("/admin")
@login_required
def admin():
    """Render the admin panel page."""
    conn = get_connection("database.db")
    users = conn.execute("SELECT id, username, role, email FROM users ORDER BY id").fetchall()
    conn.close()
    return render_template("admin.html", users=users)


@main_bp.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    """Delete a user from the admin panel. Prevent deleting primary admin or self."""
    # Only admin can delete users — verified against the DB-backed role
    # (g.current_user / users table), NOT the possibly-stale session key,
    # so valid admins no longer hit "Only admin may delete users."
    if not _current_user_is_admin():
        return jsonify({'error': 'forbidden', 'message': 'Only admin may delete users.'}), 403
    # Prevent self-delete
    if user_id == session.get('user_id'):
        return jsonify({'error': 'forbidden', 'message': 'Cannot delete your own account.'}), 400
    conn = get_connection(current_app.config['DATABASE_PATH'])
    try:
        user = conn.execute('SELECT id, username FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user:
            return jsonify({'error': 'not_found', 'message': 'User not found.'}), 404
        # Protect primary admin (id == 1)
        try:
            uid = int(user['id'])
        except Exception:
            uid = None
        if uid == 1:
            return jsonify({'error': 'forbidden', 'message': 'Cannot delete primary admin.'}), 403
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        current_app.logger.exception('Failed to delete user %s: %s', user_id, e)
        return jsonify({'error': 'delete_failed', 'message': 'Failed to delete user.'}), 500
    finally:
        conn.close()


@main_bp.route('/reports/download/<path:filename>')
@login_required
def download_report_file(filename):
    """Download a previously generated PDF report from the reports/ directory."""
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))
    safe_name = os.path.normpath(filename).replace('..', '')
    candidate = os.path.abspath(os.path.join(reports_dir, safe_name))
    # Ensure file is inside reports_dir
    if not os.path.commonpath([candidate, reports_dir]) == reports_dir or not os.path.exists(candidate):
        return jsonify({"error": "file_not_found", "message": "Report not found."}), 404
    try:
        return send_from_directory(reports_dir, os.path.basename(candidate), as_attachment=True)
    except Exception as e:
        current_app.logger.exception("Failed to send report %s: %s", candidate, e)
        return jsonify({"error": "send_failed", "message": "Failed to send report."}), 500


@main_bp.route('/reports/delete/<path:filename>', methods=['POST'])
@login_required
def delete_report_file(filename):
    """Delete a report file (admin-only).

    Returns JSON and redirects back to reports page on success.
    """
    if session.get('role') != 'admin':
        return jsonify({"error": "forbidden", "message": "Only admin may delete reports."}), 403
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))
    safe_name = os.path.normpath(filename).replace('..', '')
    candidate = os.path.abspath(os.path.join(reports_dir, safe_name))
    if not os.path.commonpath([candidate, reports_dir]) == reports_dir or not os.path.exists(candidate):
        return jsonify({"error": "file_not_found", "message": "Report not found."}), 404
    try:
        os.remove(candidate)
        flash('Report deleted', 'success')
        return redirect(url_for('main.reports'))
    except Exception as e:
        current_app.logger.exception('Failed to delete report %s: %s', candidate, e)
        return jsonify({"error": "delete_failed", "message": "Failed to delete report."}), 500


@main_bp.route('/reports/export_csv')
@login_required
def export_reports_csv():
    """Export all detection history as CSV. Admin exports all; normal users export their own records."""
    conn = get_connection(current_app.config['DATABASE_PATH'])
    try:
        if session.get('role') == 'admin':
            rows = conn.execute('SELECT * FROM detections ORDER BY created_at DESC').fetchall()
        else:
            rows = conn.execute('SELECT * FROM detections WHERE user_id = ? ORDER BY created_at DESC', (session.get('user_id'),)).fetchall()
        columns = rows[0].keys() if rows else []
        import csv
        from io import StringIO
        si = StringIO()
        writer = csv.writer(si)
        writer.writerow(columns)
        for r in rows:
            row = [r[c] for c in columns]
            writer.writerow(row)
        output = si.getvalue()
        return Response(output, mimetype='text/csv', headers={
            'Content-Disposition': 'attachment; filename=detections_export.csv'
        })
    finally:
        conn.close()


@main_bp.route("/settings")
@login_required
def settings():
    """Render the settings page."""
    conn = get_connection("database.db")
    settings_row = conn.execute("SELECT * FROM settings ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return render_template("settings.html", settings=settings_row)


@main_bp.route('/settings/upload_wallpaper', methods=['POST'])
@login_required
def upload_wallpaper():
    """Upload a wallpaper image to be used as watermark/wallpaper in dark mode.

    Saves the image under static/wallpapers/ and returns JSON {url: ...} on success.
    """
    if 'wallpaper' not in request.files:
        return jsonify({'error': 'no_file'}), 400
    file = request.files['wallpaper']
    if file.filename == '':
        return jsonify({'error': 'empty_filename'}), 400
    filename = secure_filename(file.filename)
    # Ensure extension is image-like
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ('jpg', 'jpeg', 'png', 'webp', 'gif'):
        return jsonify({'error': 'invalid_type', 'message': 'Only images allowed'}), 400
    wallpapers_dir = os.path.join(current_app.static_folder, 'wallpapers')
    os.makedirs(wallpapers_dir, exist_ok=True)
    save_path = os.path.join(wallpapers_dir, filename)
    # If filename exists, add timestamp
    if os.path.exists(save_path):
        base, dot, extpart = filename.rpartition('.')
        filename = f"{base}_{int(time.time())}.{extpart}"
        save_path = os.path.join(wallpapers_dir, filename)
    try:
        file.save(save_path)
        url = url_for('static', filename=f'wallpapers/{filename}')
        return jsonify({'url': url}), 200
    except Exception as e:
        current_app.logger.exception('Failed to save wallpaper: %s', e)
        return jsonify({'error': 'save_failed', 'message': str(e)}), 500


@main_bp.route('/settings/remove_wallpaper', methods=['POST'])
@login_required
def remove_wallpaper():
    """Remove a previously uploaded wallpaper (best-effort). Accepts JSON {url: ...} or form data.

    Even if server-side deletion fails, the client will clear local settings.
    """
    data = request.get_json(silent=True) or request.form
    url = data.get('url')
    if not url:
        return jsonify({'success': True})
    # URL is like /static/wallpapers/<name> or full path; try to resolve
    parsed = urlparse(url)
    path = parsed.path if parsed.scheme in ('http', 'https') else url
    if path.startswith('/static/'):
        rel = path[len('/static/'):]
        candidate = os.path.join(current_app.static_folder, rel)
    elif path.startswith('static/'):
        rel = path[len('static/'):]
        candidate = os.path.join(current_app.static_folder, rel)
    else:
        # attempt to strip leading / and join
        candidate = os.path.join(current_app.static_folder, path.lstrip('/'))
    try:
        if os.path.exists(candidate) and os.path.commonpath([os.path.abspath(candidate), current_app.static_folder]) == os.path.abspath(current_app.static_folder):
            os.remove(candidate)
            return jsonify({'success': True})
    except Exception as e:
        current_app.logger.exception('Failed to remove wallpaper: %s', e)
    return jsonify({'success': True})


@main_bp.route("/profile")
@login_required
def profile():
    """Render the profile page."""
    conn = get_connection("database.db")
    user = conn.execute("SELECT id, username, email, full_name, role FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()
    return render_template("profile.html", user=user)