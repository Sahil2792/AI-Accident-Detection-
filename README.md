# AI-Based Smart Road Accident Detection & Emergency Response System

This project is a production-ready Flask-based skeleton for a final year B.Tech CSE project.
It provides a clean modular structure for future YOLOv8 and OpenCV integration while keeping
all current pages and UI functional.

## Features

- Session-based authentication for admin and user login
- Dashboard with modern cards and responsive layout
- Live webcam, image upload, and video upload placeholders
- Detection history, analytics, reports, admin panel, settings, and profile pages
- SQLite storage with tables for users, detections, reports, and settings
- Scalable folder structure for future AI modules

## Project Structure

- app.py: Flask application entry point
- config.py: Global configuration values
- models/: Data models
- database/: Database setup and initialization
- detector/: Future AI detector placeholders
- alerts/: Alert module placeholders
- reports/: Report generation helpers
- utils/: Shared helpers and auth utilities
- routes/: Flask blueprints
- templates/: HTML views
- static/: CSS, JavaScript, images, uploads, and results
- tests/: Basic application tests

## Run Locally

1. Create and activate a virtual environment
2. Install dependencies: pip install -r requirements.txt
3. Start the app: python app.py
4. Open http://127.0.0.1:5000/login

## Notes

AI detection is intentionally not implemented yet. The current version focuses on architecture,
clean code organization, and UI placeholders for future YOLO integration.
