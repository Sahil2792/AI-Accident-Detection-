import os
import json
import requests
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def _alerts_store_path(app_root: str) -> str:
    folder = os.path.join(app_root, "alerts")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "sent_alerts.json")


def _load_sent_alerts(app_root: str) -> set:
    path = _alerts_store_path(app_root)
    try:
        if not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(int(x) for x in data)
    except Exception:
        # On any failure, return empty set to avoid blocking alerts
        return set()


def _save_sent_alerts(app_root: str, s: set) -> None:
    path = _alerts_store_path(app_root)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(s), f)
    except Exception:
        # Best-effort only
        pass


def _resolve_media_path(static_folder: str, app_root: str, stored_path: str | None) -> str | None:
    if not stored_path:
        return None
    normalized = str(stored_path).replace("\\", "/")
    # static/ paths
    if normalized.startswith("/static/"):
        rel = normalized[len("/static/") :]
        candidate = os.path.join(static_folder, rel)
        if os.path.exists(candidate):
            return candidate
    if normalized.startswith("static/"):
        candidate = os.path.join(app_root, normalized)
        if os.path.exists(candidate):
            return candidate
    # absolute
    if os.path.isabs(normalized) and os.path.exists(normalized):
        return normalized
    # try under static folder
    candidate = os.path.join(static_folder, normalized)
    if os.path.exists(candidate):
        return candidate
    # last resort: project-root relative
    candidate = os.path.join(app_root, normalized)
    if os.path.exists(candidate):
        return candidate
    return None


def send_telegram_alert(message_text: str, image_path: str | None = None) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        # Not configured
        return False
    try:
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            with open(image_path, "rb") as img:
                files = {"photo": img}
                data = {"chat_id": chat_id, "caption": message_text}
                resp = requests.post(url, data=data, files=files, timeout=15)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {"chat_id": chat_id, "text": message_text}
            resp = requests.post(url, data=data, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def send_email_alert(subject: str, body: str, to_addrs: list | None = None, image_path: str | None = None) -> bool:
    smtp_host = os.environ.get("ALERT_SMTP_HOST")
    smtp_port = int(os.environ.get("ALERT_SMTP_PORT", "465"))
    smtp_user = os.environ.get("ALERT_SMTP_USER")
    smtp_pass = os.environ.get("ALERT_SMTP_PASSWORD")
    from_addr = os.environ.get("ALERT_FROM_EMAIL")
    if not smtp_host or not smtp_user or not smtp_pass or not from_addr:
        return False
    to_addrs = to_addrs or [os.environ.get("ALERT_TO_EMAIL")]
    try:
        msg = MIMEMultipart()
        msg["From"] = from_addr
        msg["To"] = ", ".join([a for a in to_addrs if a])
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(image_path)}")
            msg.attach(part)

        # Use SMTP_SSL for port 465, otherwise attempt STARTTLS
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
            server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(from_addr, [a for a in to_addrs if a], msg.as_string())
        server.quit()
        return True
    except Exception:
        return False


def notify_accident(detection_id: int, detection_record: dict, static_folder: str, app_root: str) -> None:
    """Send Telegram + Email alerts for a detection. Best-effort only; failures won't raise.

    Parameters:
    - detection_id: integer id from detections table
    - detection_record: plain dict (not sqlite Row)
    - static_folder: absolute path to Flask static folder
    - app_root: application root path
    """
    try:
        # Only alert accidents
        # session_summary stored as JSON in session_summary column may contain status/confidence
        status = detection_record.get("status") or detection_record.get("detection_type")
        # If session_summary exists, prefer its status
        session_summary = detection_record.get("session_summary")
        try:
            if isinstance(session_summary, str):
                session_summary = json.loads(session_summary)
        except Exception:
            pass
        if isinstance(session_summary, dict) and session_summary.get("status"):
            status = session_summary.get("status")

        if not status or str(status).lower() != "accident":
            return

        # Deduplicate
        sent = _load_sent_alerts(app_root)
        try:
            if int(detection_id) in sent:
                return
        except Exception:
            # if cast fails, continue
            pass

        severity = detection_record.get("severity") or (session_summary.get("severity") if isinstance(session_summary, dict) else None) or "unknown"
        confidence = detection_record.get("confidence") or (session_summary.get("confidence") if isinstance(session_summary, dict) else None) or 0.0
        created_at = detection_record.get("created_at") or ""
        # fallback to local current time if missing or empty
        if not created_at:
            try:
                import datetime

                created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                created_at = ""

        media_type = detection_record.get("media_type") or detection_record.get("detection_type") or ""
        title = detection_record.get("title") or f"Detection {detection_id}"

        confidence_pct = float(confidence) * 100 if confidence is not None else 0.0

        message_text = (
            f"Accident Detected\n"
            f"Title: {title}\n"
            f"Severity: {severity}\n"
            f"Confidence: {confidence_pct:.2f}%\n"
            f"Source: {media_type}\n"
            f"Time: {created_at}"
        )

        # Resolve image/screenshot path
        screenshot_path = detection_record.get("screenshot_path") or detection_record.get("frame_path") or detection_record.get("image_path") or detection_record.get("output_video_path")
        image_abs = _resolve_media_path(static_folder, app_root, screenshot_path)

        t_ok = send_telegram_alert(message_text, image_abs)
        e_ok = send_email_alert(f"Accident Alert - {title}", message_text, image_path=image_abs)

        if t_ok or e_ok:
            try:
                sent.add(int(detection_id))
                _save_sent_alerts(app_root, sent)
            except Exception:
                pass
    except Exception:
        # never raise from alerting (best-effort)
        try:
            traceback.print_exc()
        except Exception:
            pass
