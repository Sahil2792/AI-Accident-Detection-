import os
import cv2
from ultralytics import YOLO

def get_accident_model_path():
    """Locate custom accident model weights (best (1).pt) or fallback."""
    candidates = [
        'best (1).pt',
        os.path.join('models', 'best (1).pt'),
        os.path.join('best.pt'),
        os.path.join('models', 'best.pt'),
        os.path.join('runs', 'detect', 'models', 'accident_detector', 'weights', 'best (1).pt'),
        os.path.join('runs', 'detect', 'train', 'weights', 'best (1).pt'),
        os.path.join('runs', 'detect', 'models', 'accident_detector', 'weights', 'best.pt'),
        os.path.join('runs', 'detect', 'train', 'weights', 'best.pt'),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[INFO] Custom Accident Model found at: {path}")
            return path
    
    print("[WARNING] 'best (1).pt' not found in candidate paths. Falling back to 'yolov8n.pt'")
    return 'yolov8n.pt'

def main():
    # 1. Load Both Models
    accident_model_path = get_accident_model_path()
    model_standard = YOLO('yolov8n.pt')         # Standard COCO detector (cars, persons, etc.)
    model_accident = YOLO(accident_model_path)   # Custom Accident detector

    # 2. Select Input Source
    # For Image:  SOURCE = 'test.jpg'
    # For Video:  SOURCE = 'test_video.mp4'
    # For Webcam: SOURCE = 0
    SOURCE = 'test_video.mp4'
    CONFIDENCE = 0.7

    print(f"[INFO] Running merged dual-model detection on source: '{SOURCE}'...")

    is_image = isinstance(SOURCE, str) and SOURCE.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

    if is_image:
        # --- IMAGE INFERENCE ---
        if not os.path.exists(SOURCE):
            print(f"[WARNING] Image file '{SOURCE}' not found in {os.getcwd()}")
            return

        frame = cv2.imread(SOURCE)
        if frame is not None:
            results_standard = model_standard.predict(frame, conf=CONFIDENCE, verbose=False)
            results_accident = model_accident.predict(frame, conf=CONFIDENCE, verbose=False)

            # Draw standard detections first, then overlay accident detections
            annotated_frame = results_standard[0].plot()
            annotated_frame = results_accident[0].plot(img=annotated_frame)

            cv2.imshow("Merged Detection - Image", annotated_frame)
            print("[INFO] Press any key in the window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"[ERROR] Could not load image from {SOURCE}")

    else:
        # --- VIDEO / WEBCAM INFERENCE ---
        cap = cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video/webcam source: {SOURCE}")
            return

        print("[INFO] Playing stream. Press 'q' in the window to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run predictions on current frame
            results_standard = model_standard.predict(frame, conf=CONFIDENCE, verbose=False)
            results_accident = model_accident.predict(frame, conf=CONFIDENCE, verbose=False)

            # Plot standard detections, then overlay accident predictions
            annotated_frame = results_standard[0].plot()
            annotated_frame = results_accident[0].plot(img=annotated_frame)

            # Display combined result frame
            cv2.imshow("Merged Detection - Live/Video Feed", annotated_frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Stream finished and windows closed.")

if __name__ == '__main__':
    main()