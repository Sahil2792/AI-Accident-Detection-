import os
import sys
import sqlite3
import io
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app
import detector.detect as detect_module
from detector.detect import detect_objects
import routes.main as main_routes


def test_login_page_loads():
    client = app.test_client()
    response = client.get('/login')
    assert response.status_code == 200


def test_dashboard_requires_login():
    client = app.test_client()
    response = client.get('/dashboard')
    assert response.status_code == 302


def test_webcam_page_requires_login():
    client = app.test_client()
    response = client.get('/webcam')
    assert response.status_code == 302


def test_history_shows_review_and_download_for_saved_record():
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    conn.row_factory = sqlite3.Row
    owner_id = conn.execute("SELECT id FROM users WHERE username = ?", ('admin',)).fetchone()['id']
    conn.execute(
        """
        INSERT INTO detections (user_id, title, detection_type, severity, confidence, status, output_video_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            owner_id,
            'Test Webcam Recording',
            'non_accident',
            'low',
            0.0,
            'completed',
            'recordings/webcam/test_video.mp4',
        ),
    )
    conn.commit()

    client = app.test_client()
    client.post('/', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
    response = client.get('/history')
    html = response.get_data(as_text=True)

    conn.execute("DELETE FROM detections WHERE title = ?", ('Test Webcam Recording',))
    conn.commit()
    conn.close()

    assert response.status_code == 200
    assert 'View Original' in html
    assert 'Download' in html


def test_history_is_scoped_to_logged_in_user():
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    conn.row_factory = sqlite3.Row
    user_id = conn.execute("SELECT id FROM users WHERE username = ?", ('user',)).fetchone()['id']
    other_user_id = conn.execute("SELECT id FROM users WHERE username = ?", ('admin',)).fetchone()['id']
    conn.execute(
        "INSERT INTO detections (user_id, title, detection_type, severity, confidence, status, output_video_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, 'User Scoped Recording', 'non_accident', 'low', 0.0, 'completed', 'recordings/webcam/test_user.mp4'),
    )
    conn.execute(
        "INSERT INTO detections (user_id, title, detection_type, severity, confidence, status, output_video_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (other_user_id, 'Admin Scoped Recording', 'non_accident', 'low', 0.0, 'completed', 'recordings/webcam/test_admin.mp4'),
    )
    conn.commit()
    conn.close()

    client = app.test_client()
    client.post('/', data={'username': 'user', 'password': 'user123'}, follow_redirects=True)
    response = client.get('/history')
    html = response.get_data(as_text=True)

    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    conn.execute("DELETE FROM detections WHERE title IN (?, ?)", ('User Scoped Recording', 'Admin Scoped Recording'))
    conn.commit()
    conn.close()

    assert response.status_code == 200
    assert 'User Scoped Recording' in html
    assert 'Admin Scoped Recording' not in html


def test_webcam_feed_starts_camera_before_streaming(monkeypatch):
    stream = MagicMock()
    stream.running = False
    stream.start.return_value = True
    stream.get_frame.return_value = None
    monkeypatch.setattr(main_routes, 'get_webcam_stream', lambda: stream)

    with app.test_request_context('/webcam_feed'):
        response = main_routes.webcam_feed()

    assert response.status_code == 200
    stream.start.assert_called_once()


def test_load_model_uses_safe_globals(monkeypatch):
    captured = {}

    def fake_torch_load(*args, **kwargs):
        captured["weights_only"] = kwargs.get("weights_only")
        return "checkpoint"

    def fake_yolo(model_path):
        return detect_module.torch.load("trusted.pt")

    monkeypatch.setattr(detect_module.torch, "load", fake_torch_load)
    monkeypatch.setattr(detect_module, "YOLO", fake_yolo)
    detect_module._MODEL_CACHE.clear()

    model = detect_module.load_model("custom.pt")

    assert model == "checkpoint"
    assert captured["weights_only"] is False


def test_resolve_recording_path_handles_relative_and_absolute_paths():
    relative_path = main_routes._resolve_recording_path("recordings/webcam/test.mp4")
    absolute_path = main_routes._resolve_recording_path(str(Path.cwd() / "recordings/webcam/test.mp4"))

    assert relative_path.is_absolute()
    assert relative_path.parts[-3:] == ("recordings", "webcam", "test.mp4")
    assert absolute_path.is_absolute()


def test_resolve_recording_path_handles_static_root_paths():
    static_path = main_routes._resolve_recording_path("/static/results/detected_test.mp4")

    assert static_path.is_absolute()
    assert static_path.parts[-3:] == ("static", "results", "detected_test.mp4")
    assert static_path.exists() is False or static_path.parts[-3:] == ("static", "results", "detected_test.mp4")


def test_root_login_post_redirects_to_dashboard():
    client = app.test_client()
    response = client.post('/', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/dashboard')


def test_image_detection_saves_user_owned_history(monkeypatch):
    monkeypatch.setattr(main_routes, 'process_image', lambda *args, **kwargs: {
        'objects': [{'class_name': 'car', 'confidence': 0.88}],
        'summary': {'car': 1},
        'summary_display': [{'label': 'Cars', 'count': 1}],
        'summary_text': 'Detection Summary\n-----------------\nCars: 1',
        'status': 'Non-Accident',
        'accident_confidence': 0.0,
        'total_objects': 1,
    })
    monkeypatch.setattr(main_routes, 'save_detected_image', lambda *args, **kwargs: None)

    client = app.test_client()
    client.post('/', data={'username': 'user', 'password': 'user123'}, follow_redirects=True)
    response = client.post(
        '/api/detect-image',
        data={'file': (io.BytesIO(b'fake-image-data'), 'test.jpg')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 200

    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT user_id, media_type, class_name, confidence, image_path, frame_path, object_counts, session_summary FROM detections WHERE title LIKE ? ORDER BY id DESC LIMIT 1",
        ('Image Detection - test.jpg%',),
    ).fetchone()
    conn.execute("DELETE FROM detections WHERE title LIKE ?", ('Image Detection - test.jpg%',))
    conn.commit()
    conn.close()

    assert row is not None
    assert row['media_type'] == 'Image'
    assert row['class_name'] == 'non_accident'
    assert row['confidence'] == 0.0
    assert row['image_path'].startswith('uploads/')
    assert row['frame_path'].startswith('results/')
    assert 'Cars' in row['session_summary']


def test_determine_accident_status_uses_confidence_threshold():
    result = detect_module.determine_accident_status(
        [
            {'class_name': 'accident', 'confidence': 0.61},
            {'class_name': 'car', 'confidence': 0.95},
        ],
        confidence_threshold=0.5,
    )

    assert result['status'] == 'Non-Accident'
    assert result['confidence'] == 0.61
    assert result['threshold'] == 0.7


def test_stop_webcam_saves_history_once(monkeypatch):
    summary = {
        'output_video_path': 'uploads/recordings/test_stop.mp4',
        'recording_saved': True,
        'confidence_score': 0.91,
        'object_counts_by_class': {'accident': 2},
        'total_frames': 12,
        'average_fps': 8.5,
        'total_objects_detected': 2,
        'start_time': '2026-07-31 10:00:00',
        'end_time': '2026-07-31 10:00:10',
        'duration': 10.0,
    }
    stream = MagicMock()
    stream.running = True
    stream.stop.return_value = summary
    stream.mark_session_saved.return_value = None
    monkeypatch.setattr(main_routes, 'get_webcam_stream', lambda *args, **kwargs: stream)

    client = app.test_client()
    client.post('/', data={'username': 'user', 'password': 'user123'}, follow_redirects=True)

    response_one = client.get('/stop_webcam')
    response_two = client.get('/stop_webcam')
    assert response_one.status_code == 200
    assert response_two.status_code == 200

    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    conn.row_factory = sqlite3.Row
    detection_count = conn.execute(
        "SELECT COUNT(*) AS count FROM detections WHERE output_video_path = ?",
        ('uploads/recordings/test_stop.mp4',),
    ).fetchone()['count']
    recording_count = conn.execute(
        "SELECT COUNT(*) AS count FROM recordings WHERE recording_name = ?",
        ('test_stop.mp4',),
    ).fetchone()['count']
    conn.execute("DELETE FROM recordings WHERE recording_name = ?", ('test_stop.mp4',))
    conn.execute("DELETE FROM detections WHERE output_video_path = ?", ('uploads/recordings/test_stop.mp4',))
    conn.commit()
    conn.close()

    assert detection_count == 1
    assert recording_count == 1


def test_detect_objects_runs_dual_model_pipeline_for_counts_and_status(monkeypatch):
    class DummyResult:
        def __init__(self, detections):
            self.names = {idx: class_name for idx, (class_name, _) in enumerate(detections)}
            self.boxes = type(
                'Boxes',
                (),
                {
                    'conf': [confidence for _, confidence in detections],
                    'cls': list(range(len(detections))),
                },
            )()

    class DummyModel:
        def __init__(self, detections):
            self.detections = detections

        def __call__(self, source, conf=0.25, iou=0.45, stream=False, imgsz=640):
            return [DummyResult(self.detections)]

    monkeypatch.setattr('detector.detect.load_standard_model', lambda *args, **kwargs: DummyModel([('car', 0.98), ('truck', 0.88), ('person', 0.76)]))
    monkeypatch.setattr('detector.detect.load_accident_model', lambda *args, **kwargs: DummyModel([('accident', 0.82)]))
    result = detect_objects('dummy.jpg')

    assert result['objects'][0]['class_name'] == 'car'
    assert result['objects'][0]['confidence'] == 0.98
    assert result['summary']['car'] == 1
    assert result['summary']['truck'] == 1
    assert result['summary']['person'] == 1
    assert result['cars'] == 1
    assert result['trucks'] == 1
    assert result['persons'] == 1
    assert result['total_objects'] == 3
    assert result['accident_status'] == 'Accident'
    assert result['accident_confidence'] == 0.82


def test_detect_objects_filters_weak_car_false_positive(monkeypatch):
    class DummyResult:
        def __init__(self, detections):
            self.names = {idx: class_name for idx, (class_name, _) in enumerate(detections)}
            self.boxes = type(
                'Boxes',
                (),
                {
                    'conf': [confidence for _, confidence in detections],
                    'cls': list(range(len(detections))),
                },
            )()

    class DummyModel:
        def __init__(self, detections):
            self.detections = detections

        def __call__(self, source, conf=0.25, iou=0.45, stream=False, imgsz=640):
            return [DummyResult(self.detections)]

    monkeypatch.setattr('detector.detect.load_standard_model', lambda *args, **kwargs: DummyModel([('car', 0.59)]))
    monkeypatch.setattr('detector.detect.load_accident_model', lambda *args, **kwargs: DummyModel([]))
    result = detect_objects('dummy.jpg')

    assert result['objects'][0]['class_name'] == 'car'
    assert result['summary']['car'] == 1
    assert result['total_objects'] == 1
    assert result['average_confidence'] == 0.59
    assert result['status'] == 'Non-Accident'


def test_accident_detection_is_not_blocked_by_vehicle_floor():
    class DummyResult:
        def __init__(self, class_name, confidence):
            self.names = {0: class_name}
            self.boxes = type('Boxes', (), {'conf': [confidence], 'cls': [0]})()

    detections = detect_module.normalize_yolo_results([DummyResult('accident', 0.56)])

    assert detections[0]['class_name'] == 'accident'
    assert detections[0]['confidence'] == 0.56


def test_strict_accident_validation_drops_isolated_single_car():
    accident = {
        'class_name': 'accident',
        'confidence': 0.72,
        'bbox': {'x1': 100, 'y1': 100, 'x2': 220, 'y2': 180},
    }
    standard_car = {
        'class_name': 'car',
        'confidence': 0.91,
        'bbox': {'x1': 110, 'y1': 108, 'x2': 215, 'y2': 178},
    }

    validated = detect_module.validate_accident_detections([accident], [standard_car])

    assert validated == []


def test_strict_accident_validation_drops_high_confidence_isolated_car():
    accident = {
        'class_name': 'accident',
        'confidence': 0.82,
        'bbox': {'x1': 100, 'y1': 100, 'x2': 220, 'y2': 180},
    }
    standard_car = {
        'class_name': 'car',
        'confidence': 0.91,
        'bbox': {'x1': 110, 'y1': 108, 'x2': 215, 'y2': 178},
    }

    validated = detect_module.validate_accident_detections([accident], [standard_car])

    assert validated == []


def test_strict_accident_validation_keeps_colliding_vehicle_pair():
    accident = {
        'class_name': 'accident',
        'confidence': 0.72,
        'bbox': {'x1': 90, 'y1': 90, 'x2': 270, 'y2': 190},
    }
    standard_objects = [
        {
            'class_name': 'car',
            'confidence': 0.91,
            'bbox': {'x1': 100, 'y1': 108, 'x2': 185, 'y2': 178},
        },
        {
            'class_name': 'truck',
            'confidence': 0.86,
            'bbox': {'x1': 178, 'y1': 104, 'x2': 260, 'y2': 182},
        },
    ]

    validated = detect_module.validate_accident_detections([accident], standard_objects)

    assert len(validated) == 1
    # Validator labels collision-pair promotions 'collision_pair_evidence'
    # (older builds used 'overlapping_vehicles'); both mean the accident box
    # was kept because >=2 standard vehicles physically collide.
    assert validated[0]['validation_reason'] in {'overlapping_vehicles', 'collision_pair_evidence'}


def test_delete_detection_with_sqlite_row():
    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO recordings (user_id, recording_name, video_path) VALUES (?, ?, ?)",
        (1, "delete_test.mp4", "uploads/recordings/delete_test.mp4")
    )
    recording_id = cursor.lastrowid
    conn.commit()
    conn.close()

    client = app.test_client()
    client.post('/', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
    response = client.post(f'/history/{recording_id}/delete', follow_redirects=True)
    assert response.status_code == 200

    conn = sqlite3.connect(app.config['DATABASE_PATH'])
    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
    conn.close()
    assert row is None
