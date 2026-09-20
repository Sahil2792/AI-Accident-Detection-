"""Simple PDF report generator.

This module uses reportlab (add to requirements.txt) to create a small PDF
containing the detection summary, status, confidence, severity and an
embedded screenshot when available. It is intentionally minimal and
robust: failure to embed a screenshot will still produce a textual PDF.
"""

import os
import time
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _safe_image_path(path: str) -> str | None:
    if not path:
        return None
    candidate = path.replace('\\', '/')
    # If it's already under static/, try that absolute path
    if candidate.startswith('/static/') or candidate.startswith('static/'):
        static_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
        rel = candidate.split('/static/', 1)[-1]
        full = os.path.join(static_root, rel)
        if os.path.exists(full):
            return full
    # If absolute path exists, use it
    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate
    # Try relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full = os.path.join(project_root, candidate)
    if os.path.exists(full):
        return full
    return None


def generate_pdf_report(data: dict) -> str:
    """Generate a PDF report for a single detection record.

    Args:
        data: dict containing keys: title, status, confidence, severity, screenshot_path,
              summary_text (optional), created_at (optional)

    Returns:
        Absolute path to generated PDF file.
    """
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"detection_report_{timestamp}.pdf"
    out_path = os.path.join(reports_dir, filename)

    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4

    x_margin = 40
    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, data.get('title', 'Detection Report'))

    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(x_margin, y, f"Status: {data.get('status', 'Unknown')}")
    y -= 18
    conf = float(data.get('confidence', 0.0))
    c.drawString(x_margin, y, f"Confidence: {conf * 100:.2f}%")
    y -= 18
    c.drawString(x_margin, y, f"Severity: {data.get('severity', 'low').title()}")
    y -= 24

    created_at = data.get('created_at')
    if created_at:
        c.drawString(x_margin, y, f"Recorded at: {created_at}")
        y -= 20

    # Summary text (multi-line)
    summary_text = data.get('summary_text') or ''
    if summary_text:
        # limit summary area height
        max_summary_height = 200
        text_obj = c.beginText(x_margin, y)
        text_obj.setFont('Helvetica', 10)
        for line in str(summary_text).splitlines():
            text_obj.textLine(line)
            y -= 12
            if (height - y) > max_summary_height:
                text_obj.textLine('...')
                break
        c.drawText(text_obj)
        y -= 10

    # Screenshot embedding (if available)
    screenshot = data.get('screenshot_path') or data.get('screenshot')
    image_file = _safe_image_path(screenshot) if screenshot else None
    if image_file:
        try:
            # Fit image into a box on the right half of the page
            max_w = (width - (2 * x_margin)) * 0.9
            max_h = 300
            img = ImageReader(image_file)
            iw, ih = img.getSize()
            scale = min(max_w / iw, max_h / ih, 1.0)
            draw_w, draw_h = iw * scale, ih * scale
            # Position image centered horizontally
            img_x = x_margin
            img_y = y - draw_h - 10
            if img_y < 60:
                c.showPage()
                y = height - 40
                img_y = y - draw_h - 10
            c.drawImage(img, img_x, img_y, width=draw_w, height=draw_h)
            y = img_y - 20
        except Exception:
            # If embedding fails, fall back to writing the path.
            c.drawString(x_margin, y, f"Screenshot: {image_file}")
            y -= 18

    c.showPage()
    c.save()
    return out_path
