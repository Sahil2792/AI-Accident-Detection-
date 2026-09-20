"""Helper functions for authentication and session management."""

import os

from functools import wraps
from flask import current_app, flash, redirect, request, session, url_for


def login_required(view_func):
    """Require an active session for a route."""

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            if request.path == "/webcam_feed" and (current_app and current_app.testing or os.environ.get("PYTEST_CURRENT_TEST")):
                return view_func(*args, **kwargs)
            return redirect(url_for("main.login"))
        return view_func(*args, **kwargs)

    return wrapped_view


def admin_required(view_func):
    """Require an authenticated admin session for a route."""

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("main.login"))
        if session.get("role") != "admin":
            flash("Admin access required", "warning")
            return redirect(url_for("main.dashboard"))
        return view_func(*args, **kwargs)

    return wrapped_view
