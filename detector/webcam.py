"""Real-time universal camera stream processing with native MJPEG reader, fail-safe fallback, and dual YOLO model inference."""

import os
import time
import urllib.request
from collections import deque
from datetime import datetime
from queue import Empty, Full, Queue
from threading import Lock, Thread, current_thread

import cv2
import numpy as np

from detector.detect import (
    ACCIDENT_CLASSES,
    ACCIDENT_INFERENCE_CONFIDENCE,
    ACCIDENT_MIN_CONFIDENCE,
    INFERENCE_IOU,
    INFERENCE_SIZE,
    STANDARD_INFERENCE_CONFIDENCE,
    STANDARD_OBJECT_CLASSES,
    combine_detections_with_accident_priority,
    derive_status,
    get_accident_confidence,
    get_detection_color,
    load_accident_model,
    load_standard_model,
    normalize_yolo_results,
    resolve_model_path,
    summarize_detections,
    validate_accident_detections,
)
from detector.video import (
    ACCIDENT_MAX_FRAME_GAP,
    ACCIDENT_MIN_CONSECUTIVE_FRAMES,
    ACCIDENT_VIDEO_CONFIDENCE,
    _TemporalAccidentConfirmer,
)
from detector.inference_robust import (
    REALTIME_CONF,
    REALTIME_ENHANCE,
    REALTIME_IMGSZ,
    REALTIME_IOU,
    REALTIME_MULTI_SCALE_SIZES,
    REALTIME_STANDARD_CONF,
    REALTIME_TTA,
    run_robust_inference,
)
from detector.severity import classify_severity
from detector.screenshot import save_frame_screenshot

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEBCAM_VIDEO_DIR = os.path.join(PROJECT_ROOT, "uploads", "recordings")
FRAME_SIZE = (640, 480)
DETECTION_INTERVAL = 2
RECORDING_QUEUE_SIZE = 60
RECORDING_WARMUP_FRAMES = 8
RECORDING_BUFFER_LIMIT = 120
MIN_RECORDING_FPS = 3.0

# Number of inference frames to process before accident status is reported.
# Prevents the webcam from instantly triggering an "accident" on the very
# first frame or upon initialization. Standard object detections (car,
# person, truck, etc.) are always shown immediately.
WEBCAM_ACCIDENT_WARMUP_FRAMES = int(os.environ.get("WEBCAM_ACCIDENT_WARMUP_FRAMES", "3"))

CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", 0)


class MJPEGStreamCapture:
    """Native Python HTTP MJPEG stream capture for IP webcams."""

    def __init__(self, url):
        self.url = url
        self.stream = None
        self.bytes_data = b""
        self.last_error = None

    def isOpened(self):
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "Mozilla/5.0"})
            res = urllib.request.urlopen(req, timeout=4)
            first_chunk = res.read(1024)
            self.stream = res
            self.bytes_data = first_chunk
            return True
        except Exception as err:
            self.last_error = f"Stream connection failed: {err}"
            print(f"[DEBUG] MJPEGStreamCapture failed for {self.url}: {err}")
            return False

    def read(self):
        if not self.stream:
            return False, None
        try:
            for _ in range(200):
                a = self.bytes_data.find(b"\xff\xd8")
                if a != -1:
                    b = self.bytes_data.find(b"\xff\xd9", a + 2)
                    if b != -1:
                        jpg_bytes = self.bytes_data[a : b + 2]
                        self.bytes_data = self.bytes_data[b + 2 :]
                        frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            return True, frame

                chunk = self.stream.read(4096)
                if not chunk:
                    return False, None
                self.bytes_data += chunk
                if len(self.bytes_data) > 1000000:
                    self.bytes_data = b""
        except Exception as err:
            print(f"[DEBUG] Error reading frame: {err}")
            return False, None
        return False, None

    def set(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return 0

    def release(self):
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        self.stream = None


def parse_camera_source(source_val):
    """Normalize input camera source into an integer index or clean String URL."""
    if source_val is None or str(source_val).strip() == "":
        source_val = os.environ.get("CAMERA_SOURCE", 0)

    if isinstance(source_val, str):
        s_clean = source_val.strip()
        if s_clean.isdigit():
            return int(s_clean)

        # Format IP camera URLs
        if not s_clean.startswith("http://") and not s_clean.startswith("https://") and not s_clean.startswith("rtsp://"):
            if ":" in s_clean or "." in s_clean:
                s_clean = "http://" + s_clean

        return s_clean

    try:
        return int(source_val)
    except (ValueError, TypeError):
        return 0


def generate_placeholder_frame(message="Camera Offline / Connecting..."):
    """Generate a JPEG error placeholder frame to prevent web UI stream crashes."""
    try:
        img = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (FRAME_SIZE[0] - 20, FRAME_SIZE[1] - 20), (50, 50, 50), 2)
        cv2.putText(img, "AI ACCIDENT DETECTION SYSTEM", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, message, (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(img, "Check camera status & close other tabs/apps", (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        ret, buffer = cv2.imencode(".jpg", img)
        if ret:
            return buffer.tobytes()
    except Exception:
        pass
    return b""


def create_universal_capture(source=None):
    """
    Safely open a video capture device from an integer index (0, 1, 2) or String URL
    (e.g., IP camera URL).
    Uses native MJPEG HTTP reader for MJPEG URLs and OpenCV for camera indices.
    """
    requested_source = parse_camera_source(source)

    # 1. Primary Attempt
    cap = _try_open_source(requested_source)
    if cap and cap.isOpened():
        print(f"[INFO] Universal Camera Loader connected to source: {requested_source}")
        return cap

    print(f"[WARNING] Primary camera '{requested_source}' failed. Initiating fallback search...")

    # 2. Auto-fallback search across camera indices [0, 1, 2, 3] if numeric or failed stream
    for idx in range(4):
        if idx == requested_source:
            continue
        cap = _try_open_source(idx)
        if cap and cap.isOpened():
            print(f"[INFO] Auto-fallback camera search succeeded on index: {idx}")
            return cap

    print("[ERROR] All camera sources and fallback indices failed.")
    return None


def _try_open_source(source):
    """Attempt opening a single camera index or IP stream URL safely."""
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")):
        urls_to_try = [source]
        for url in urls_to_try:
            # First try native MJPEG HTTP capture for HTTP video streams reliably
            mjpeg_cap = MJPEGStreamCapture(url)
            if mjpeg_cap.isOpened():
                print(f"[INFO] Native MJPEG stream connected successfully via: {url}")
                return mjpeg_cap

            # Fallback to OpenCV VideoCapture
            try:
                cap = cv2.VideoCapture(url)
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[INFO] OpenCV VideoCapture stream connected via: {url}")
                        return cap
                    cap.release()
            except Exception as err:
                print(f"[DEBUG] Failed to open OpenCV stream URL {url}: {err}")
        return None

    # Numeric camera index
    try:
        index = int(source)
    except (ValueError, TypeError):
        return None

    # Try Windows CAP_DSHOW backend first
    if hasattr(cv2, "CAP_DSHOW"):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                cap.release()
        except Exception:
            pass

    # Default OpenCV backend fallback
    try:
        cap = cv2.VideoCapture(index)
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
    except Exception:
        pass

    return None


class WebcamStream:
    """Manages real-time camera capture and dual YOLO model detection."""

    def __init__(self, user_id, model_path=None, camera_source=None, output_dir=None):
        """Initialize webcam stream manager."""
        self.user_id = user_id
        self.model_path = resolve_model_path(model_path)
        self.camera_source = parse_camera_source(camera_source)
        # Dual-model pipeline: standard YOLO (yolov8n.pt) for object counting
        # + custom accident model (best.pt) for genuine accident detection.
        self.standard_model = load_standard_model()
        self.accident_model = load_accident_model(self.model_path)
        self.output_dir = output_dir or DEFAULT_WEBCAM_VIDEO_DIR
        self.cap = None
        self.capture_thread = None
        self.capture_queue = Queue(maxsize=1)
        self.video_writer = None
        self.write_queue = None
        self.writer_thread = None
        self.detection_thread = None
        self.running = False
        self.camera_available = False
        self.frame_buffer = None
        self.frame_lock = Lock()
        self.session_lock = Lock()
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        self.start_datetime = None
        self.end_time = None
        self.end_datetime = None
        self.total_objects = 0
        self.detection_summary = {}
        self.last_detections = []
        self.output_video_path = None
        self.session_summary = None
        self.current_detection_id = None
        self._session_finalized = False
        self._last_fps_time = None
        self._fps_frame_count = 0
        self.last_error = None
        self.confidence_score = 0.0
        self.accident_confidence = 0.0
        self.screenshot_path = None
        self._accident_confirmed = False
        self._accident_screenshot_saved = False
        self._capture_frame_count = 0
        self._capture_failures = 0
        self._pending_recording_frames = deque()
        self._recording_started_at = None
        self._recording_fps = None
        self._video_writer_ready = False
        # Temporal accident confirmer ensures an accident is only flagged after
        # it has been sustained across multiple consecutive inference frames.
        # This eliminates single-frame false positives on normal moving cars.
        self._accident_confirmer = _TemporalAccidentConfirmer(
            min_consecutive_frames=ACCIDENT_MIN_CONSECUTIVE_FRAMES,
            max_frame_gap=ACCIDENT_MAX_FRAME_GAP,
        )
        self._inference_frame_index = 0

    def start(self):
        """Start camera capture and background detection thread cleanly."""
        if self.running:
            return True

        self.last_error = None
        self.cap = create_universal_capture(self.camera_source)

        if self.cap is None or not self.cap.isOpened():
            error_msg = getattr(self.cap, "last_error", None) or f"Searching for camera '{self.camera_source}'..."
            self.last_error = error_msg
            print(f"[WARNING] {self.last_error}")
            self.camera_available = False
            self.running = True
            with self.frame_lock:
                self.frame_buffer = generate_placeholder_frame(error_msg)
            self.detection_thread = Thread(target=self._placeholder_loop, daemon=True)
            self.detection_thread.start()
            return True

        self.camera_available = True
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
            if hasattr(cv2, "CAP_PROP_FPS"):
                self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_video_path = os.path.join(self.output_dir, f"user_{self.user_id}_{timestamp}.mp4")

        self.running = True
        self.write_queue = None
        self.writer_thread = None
        self.frame_count = 0
        self.fps = 0
        self.total_objects = 0
        self.detection_summary = {}
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        self.end_time = None
        self.end_datetime = None
        self.last_detections = []
        self.frame_buffer = None
        self.session_summary = None
        self.current_detection_id = None
        self._session_finalized = False
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        self.confidence_score = 0.0
        self.accident_confidence = 0.0
        self.screenshot_path = None
        self._accident_confirmed = False
        self._accident_screenshot_saved = False
        self._capture_frame_count = 0
        self._capture_failures = 0
        self._pending_recording_frames.clear()
        self._recording_started_at = None
        self._recording_fps = None
        self._video_writer_ready = False
        # Reset temporal accident confirmation state for the new session.
        self._accident_confirmer = _TemporalAccidentConfirmer(
            min_consecutive_frames=ACCIDENT_MIN_CONSECUTIVE_FRAMES,
            max_frame_gap=ACCIDENT_MAX_FRAME_GAP,
        )
        self._inference_frame_index = 0

        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.detection_thread = Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        return True

    def stop(self):
        """Stop camera stream, finalize recording, and return session summary."""
        if not self.running and self.session_summary:
            return self.session_summary

        self.running = False
        if self.detection_thread and self.detection_thread.is_alive() and current_thread() is not self.detection_thread:
            self.detection_thread.join(timeout=5)

        self._release_resources()
        return self._finalize_session()

    def _placeholder_loop(self):
        """Periodic loop to keep streaming placeholder frame while attempting auto-reconnect."""
        last_retry = time.time()
        while self.running:
            with self.frame_lock:
                msg = self.last_error or f"Searching for Camera '{self.camera_source}'..."
                self.frame_buffer = generate_placeholder_frame(msg)

            time.sleep(0.5)
            # Re-attempt connecting every 3 seconds
            if time.time() - last_retry >= 3.0:
                last_retry = time.time()
                cap = create_universal_capture(self.camera_source)
                if cap and cap.isOpened():
                    print(f"[INFO] Auto-reconnect succeeded for source: {self.camera_source}")
                    self.cap = cap
                    self.camera_available = True
                    self.capture_thread = Thread(target=self._capture_loop, daemon=True)
                    self.capture_thread.start()
                    self._detection_loop()
                    break

    def _capture_loop(self):
        """Continuously pull frames from the camera so stale frames do not pile up."""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._capture_failures += 1
                if self._capture_failures > 30:
                    self.last_error = "Camera feed lost or disconnected."
                    print(f"[WARNING] {self.last_error}")
                    break
                time.sleep(0.01)
                continue

            self._capture_failures = 0
            self._capture_frame_count += 1
            frame_timestamp = time.time()

            if self.capture_queue.full():
                try:
                    self.capture_queue.get_nowait()
                except Empty:
                    pass

            try:
                self.capture_queue.put_nowait((self._capture_frame_count, frame_timestamp, frame))
            except Full:
                pass

        self.running = False

    def _detection_loop(self):
        """Continuous dual-model detection loop running in background thread."""
        try:
            while self.running:
                try:
                    capture_id, frame_timestamp, frame = self.capture_queue.get(timeout=0.2)
                except Empty:
                    if self.last_error:
                        with self.frame_lock:
                            self.frame_buffer = generate_placeholder_frame("Camera Disconnected")
                    continue

                if frame is None:
                    continue

                if frame.shape[1] != FRAME_SIZE[0] or frame.shape[0] != FRAME_SIZE[1]:
                    frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)

                self.frame_count += 1
                should_run_inference = self.frame_count % DETECTION_INTERVAL == 0

                if should_run_inference:
                    self._inference_frame_index += 1
                    try:
                        # Step 1: Standard YOLO (yolov8n.pt) counts everyday objects.
                        # REALTIME inference: lightweight imgsz=640, TTA disabled
                        # (augment=False) and NO multi-scale so the webcam stream
                        # does not lag or freeze.
                        standard_raw = run_robust_inference(
                            self.standard_model,
                            frame,
                            conf=REALTIME_STANDARD_CONF,
                            iou=REALTIME_IOU,
                            imgsz=REALTIME_IMGSZ,
                            augment=REALTIME_TTA,
                            enhance=REALTIME_ENHANCE,
                            multi_scale_sizes=list(REALTIME_MULTI_SCALE_SIZES),
                        )
                        standard_results = list([] if standard_raw is None else standard_raw)
                        standard_detections = normalize_yolo_results(
                            standard_results,
                            allowed_classes=STANDARD_OBJECT_CLASSES,
                            source_model="yolov8n.pt",
                        )

                        # Step 2: Custom accident model (best.pt).
                        # REALTIME inference: lightweight imgsz=640, TTA disabled
                        # (augment=False) and NO multi-scale so the webcam stream
                        # runs at a smooth frame rate.
                        accident_raw = run_robust_inference(
                            self.accident_model,
                            frame,
                            conf=REALTIME_CONF,
                            iou=REALTIME_IOU,
                            imgsz=REALTIME_IMGSZ,
                            augment=REALTIME_TTA,
                            enhance=REALTIME_ENHANCE,
                            multi_scale_sizes=list(REALTIME_MULTI_SCALE_SIZES),
                        )
                        accident_results = list([] if accident_raw is None else accident_raw)
                        raw_accident_detections = normalize_yolo_results(
                            accident_results,
                            allowed_classes=ACCIDENT_CLASSES,
                            source_model=os.path.basename(self.model_path),
                        )

                        # Step 3: STRICT multi-vehicle collision validation - identical rules to the
                        # image/video pipelines. An accident box survives ONLY when
                        # it overlaps >=2 DISTINCT vehicles that collide with EACH
                        # OTHER and confidence >= ACCIDENT_MIN_CONFIDENCE (0.65).
                        # This permanently stops the webcam from flagging single
                        # moving cars / shadows / roadside objects as accidents.
                        # Pure geometry on existing boxes -> no FPS cost.
                        accident_detections = [
                            detection
                            for detection in raw_accident_detections
                            if float(detection.get("confidence", 0.0)) >= ACCIDENT_VIDEO_CONFIDENCE
                        ]
                        accident_detections = validate_accident_detections(
                            accident_detections,
                            standard_detections,
                            min_confidence=ACCIDENT_MIN_CONFIDENCE,
                        )

                        # Step 4: Combine detections with accident boxes taking priority.
                        detections = combine_detections_with_accident_priority(
                            standard_detections,
                            accident_detections,
                        )

                        # Temporal accident confirmation: an accident is only
                        # accepted after it has been detected in at least
                        # ACCIDENT_MIN_CONSECUTIVE_FRAMES inference frames.
                        # A single-frame false positive (normal car briefly
                        # triggering the accident model) is dropped.
                        has_accident_this_frame = any(
                            str(d.get("class_name", "")).lower() == "accident" for d in detections
                        )
                        accident_confirmed = self._accident_confirmer.update(
                            has_accident_this_frame, self._inference_frame_index
                        )

                        # WARMUP PERIOD: Do NOT report any accident status until
                        # at least WEBCAM_ACCIDENT_WARMUP_FRAMES inference frames
                        # have been processed. This prevents the webcam from
                        # instantly triggering an "accident" on the very first
                        # frame or upon initialization. The status should only
                        # update based on actual live inference from current frames.
                        warmup_complete = self._inference_frame_index >= WEBCAM_ACCIDENT_WARMUP_FRAMES

                        # Only keep accident detections that have been temporally
                        # confirmed AND the warmup period has passed. Standard
                        # object detections are always kept.
                        detections = [
                            d for d in detections
                            if str(d.get("class_name", "")).lower() != "accident"
                            or (accident_confirmed and warmup_complete)
                        ]

                        # Track confirmed accident state for auto-screenshot.
                        # Only a freshly confirmed (and warmup-complete) accident
                        # can save a screenshot; once it stops being confirmed,
                        # the guard resets so the next event can save its own.
                        self._accident_confirmed = (
                            accident_confirmed and warmup_complete and has_accident_this_frame
                        )
                        if not self._accident_confirmed:
                            self._accident_screenshot_saved = False

                        self.last_detections = detections
                        if detections:
                            confidence_values = [d["confidence"] for d in detections]
                            self.confidence_score = round(sum(confidence_values) / len(confidence_values), 4)
                            self.accident_confidence = round(get_accident_confidence(detections), 4)
                            self.total_objects += len(detections)
                            for d in detections:
                                class_name = d["class_name"]
                                self.detection_summary[class_name] = self.detection_summary.get(class_name, 0) + 1
                    except Exception as exc:
                        self.last_error = str(exc)
                        print(f"Webcam inference error: {exc}")

                if self.last_detections:
                    self._draw_detections(frame, self.last_detections)

                # Auto-screenshot: persist the confirmed accident frame once per
                # event (dedup) after the red 'accident' box has been drawn.
                if self._accident_confirmed and not self._accident_screenshot_saved:
                    saved = save_frame_screenshot(frame, PROJECT_ROOT)
                    if saved is not None:
                        self.screenshot_path = saved
                        self._accident_screenshot_saved = True

                self._update_current_fps()
                self._draw_fps(frame)
                self._queue_recording_frame(frame, frame_timestamp)

                ret_encode, buffer = cv2.imencode(".jpg", frame)
                if ret_encode:
                    with self.frame_lock:
                        self.frame_buffer = buffer.tobytes()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"Webcam stream processing error: {exc}")
        finally:
            self.running = False
            self._release_resources()
            self._finalize_session()

    def _initialize_video_writer(self, frame):
        """Create a VideoWriter sized to the captured frame."""
        if self.video_writer or not self.output_video_path:
            return

        height, width = frame.shape[:2]
        output_fps = self._estimate_recording_fps()
        if not output_fps or output_fps < MIN_RECORDING_FPS:
            source_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0
            measured_fps = self.fps if self.fps and self.fps >= MIN_RECORDING_FPS else 0
            output_fps = source_fps if source_fps and source_fps >= MIN_RECORDING_FPS else measured_fps or 10.0

        for codec_code in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec_code)
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, output_fps, (width, height))
            if self.video_writer and self.video_writer.isOpened():
                break
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

        if not self.video_writer or not self.video_writer.isOpened():
            if self.video_writer:
                self.video_writer.release()
            self.video_writer = None
            self.last_error = f"Recording initialization failed for {self.output_video_path}"
            return

        self.write_queue = Queue(maxsize=RECORDING_QUEUE_SIZE)
        self.writer_thread = Thread(target=self._video_writer_loop, daemon=True)
        self.writer_thread.start()
        self._video_writer_ready = True

    def _estimate_recording_fps(self):
        """Estimate an output FPS from recently observed frame arrival times."""
        if len(self._pending_recording_frames) < 3:
            return self._recording_fps or 0.0

        timestamps = [item[0] for item in self._pending_recording_frames]
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return self._recording_fps or 0.0

        estimated_fps = (len(timestamps) - 1) / duration
        self._recording_fps = round(max(estimated_fps, MIN_RECORDING_FPS), 2)
        return self._recording_fps

    def _flush_pending_recording_frames(self):
        """Push buffered frames into the async writer in capture order."""
        if not self.video_writer or not self.write_queue:
            return

        while self._pending_recording_frames:
            _, frame = self._pending_recording_frames.popleft()
            self._enqueue_frame_for_writer(frame)

    def _enqueue_frame_for_writer(self, frame):
        """Enqueue a frame for the background recording writer without blocking."""
        if not self.write_queue:
            return

        frame_copy = frame.copy() if hasattr(frame, "copy") else frame
        try:
            self.write_queue.put_nowait(frame_copy)
        except Full:
            try:
                self.write_queue.get_nowait()
                self.write_queue.task_done()
            except Empty:
                pass
            try:
                self.write_queue.put_nowait(frame_copy)
            except Full:
                pass

    def _queue_recording_frame(self, frame, frame_timestamp):
        """Queue a frame for asynchronous recording without blocking display."""
        if frame is None or not self.output_video_path:
            return

        if self._recording_started_at is None:
            self._recording_started_at = frame_timestamp

        self._pending_recording_frames.append((frame_timestamp, frame.copy()))
        while len(self._pending_recording_frames) > RECORDING_BUFFER_LIMIT:
            self._pending_recording_frames.popleft()

        if not self.video_writer:
            elapsed = frame_timestamp - self._recording_started_at
            if len(self._pending_recording_frames) >= RECORDING_WARMUP_FRAMES and elapsed >= 0.5:
                self._initialize_video_writer(frame)
                self._flush_pending_recording_frames()
            return

        self._enqueue_frame_for_writer(frame)

    def _video_writer_loop(self):
        """Write queued frames asynchronously on a background thread."""
        while self.running or (self.write_queue and not self.write_queue.empty()):
            try:
                frame = self.write_queue.get(timeout=0.1)
            except Empty:
                continue

            if self.video_writer and frame is not None:
                self.video_writer.write(frame)
            if self.write_queue:
                self.write_queue.task_done()

    def _update_current_fps(self):
        """Update FPS rate calculation."""
        self._fps_frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time if self._last_fps_time else 0
        if elapsed >= 0.5:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_time = now

    def _draw_fps(self, frame):
        """Draw FPS text on frame."""
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    def _release_resources(self):
        """Release camera and recording handles cleanly."""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.frame_buffer = None

        if self.capture_thread and self.capture_thread.is_alive() and current_thread() is not self.capture_thread:
            try:
                self.capture_thread.join(timeout=3)
            except Exception:
                pass
        self.capture_thread = None
        self.capture_queue = Queue(maxsize=1)

        try:
            if self.write_queue:
                while not self.write_queue.empty():
                    try:
                        frame = self.write_queue.get_nowait()
                        if self.video_writer and frame is not None:
                            self.video_writer.write(frame)
                    except Empty:
                        break
        except Exception:
            pass

        if self.writer_thread and self.writer_thread.is_alive() and current_thread() is not self.writer_thread:
            try:
                self.writer_thread.join(timeout=3)
            except Exception:
                pass
        self.writer_thread = None
        self.write_queue = None

        if self.video_writer:
            try:
                self.video_writer.release()
            except Exception:
                pass
        self.video_writer = None
        self._video_writer_ready = False
        self._pending_recording_frames.clear()

    def _finalize_session(self):
        """Build JSON-friendly summary for the completed stream session."""
        with self.session_lock:
            if self._session_finalized:
                return self.session_summary

            self.end_time = time.time()
            self.end_datetime = datetime.now()
            duration = max((self.end_time - self.start_time), 0) if self.start_time else 0
            average_fps = self.frame_count / duration if duration else 0
            output_path = self._display_output_path()

            recording_saved = False
            if output_path and self.output_video_path and os.path.exists(self.output_video_path):
                if os.path.getsize(self.output_video_path) > 0:
                    cap = cv2.VideoCapture(self.output_video_path)
                    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                        recording_saved = True
                    cap.release()

            # If the warmup period was not completed, force the status to
            # "Non-Accident" so a session that ended before enough inference
            # frames were processed never reports a false accident.
            warmup_complete = self._inference_frame_index >= WEBCAM_ACCIDENT_WARMUP_FRAMES
            final_status = derive_status(
                self.detection_summary,
                self.accident_confidence,
                confidence_threshold=ACCIDENT_VIDEO_CONFIDENCE,
            )
            if not warmup_complete and final_status == "Accident":
                final_status = "Non-Accident"
                print(
                    f"[INFO] Webcam session ended before warmup ({self._inference_frame_index}/{WEBCAM_ACCIDENT_WARMUP_FRAMES} "
                    "inference frames). Forcing status to Non-Accident."
                )

            self.session_summary = {
                "start_time": self._format_datetime(self.start_datetime),
                "end_time": self._format_datetime(self.end_datetime),
                "duration": round(duration, 2),
                "total_frames": self.frame_count,
                "average_fps": round(average_fps, 2),
                "total_objects_detected": self.total_objects,
                "object_counts_by_class": dict(self.detection_summary),
                "status": final_status,
                "confidence_score": self.confidence_score,
                "accident_confidence": self.accident_confidence,
                "severity": classify_severity(self.accident_confidence, final_status),
                "screenshot_path": self.screenshot_path,
                "output_video_path": output_path,
                "recording_saved": recording_saved,
                "error": self.last_error,
            }
            if self.current_detection_id is not None:
                self.session_summary["detection_id"] = self.current_detection_id
            self._session_finalized = True
            return self.session_summary

    def _display_output_path(self):
        """Return relative output path."""
        if not self.output_video_path:
            return None
        return os.path.relpath(self.output_video_path, PROJECT_ROOT).replace(os.sep, "/")

    @staticmethod
    def _format_datetime(value):
        """Format datetimes consistently."""
        return value.strftime("%Y-%m-%d %H:%M:%S") if value else None

    def _draw_detections(self, frame, detections):
        """Draw bounding boxes and class labels on frame."""
        for detection in detections:
            bbox = detection.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            label = f"{class_name} ({confidence:.2%})"

            color = get_detection_color(detection)
            thickness = 3 if class_name == "accident" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return frame

    def get_frame(self):
        """Get current frame as JPEG bytes or placeholder image."""
        with self.frame_lock:
            if self.frame_buffer is not None:
                return self.frame_buffer
            return generate_placeholder_frame(f"Connecting to Camera Source '{self.camera_source}'...")

    def get_stats(self):
        """Get current detection statistics."""
        return {
            "running": self.running,
            "camera_available": self.camera_available,
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "total_objects": self.total_objects,
            "summary": self.detection_summary,
            "error": self.last_error,
        }

    def mark_session_saved(self, detection_id):
        """Attach database id to finalized stream session."""
        with self.session_lock:
            self.current_detection_id = detection_id
            if self.session_summary:
                self.session_summary["detection_id"] = detection_id


_webcam_instances = {}


def get_webcam_stream(user_id, model_path=None, camera_source=None):
    """Get or create a user-scoped webcam stream instance."""
    global _webcam_instances
    stream = _webcam_instances.get(user_id)
    requested_model_path = resolve_model_path(model_path)
    if stream is None:
        stream = WebcamStream(user_id, model_path=requested_model_path, camera_source=camera_source)
        _webcam_instances[user_id] = stream
        return stream

    if stream.model_path != requested_model_path or (
        camera_source is not None and str(stream.camera_source) != str(parse_camera_source(camera_source))
    ):
        try:
            stream.stop()
        finally:
            _webcam_instances[user_id] = WebcamStream(user_id, model_path=requested_model_path, camera_source=camera_source)

    return _webcam_instances[user_id]


def clear_webcam_stream(user_id):
    """Remove a cached webcam stream for a user."""
    global _webcam_instances
    stream = _webcam_instances.pop(user_id, None)
    if stream:
        try:
            stream.stop()
        except Exception:
            pass
