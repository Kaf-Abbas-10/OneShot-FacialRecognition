from deepface import DeepFace
import cv2
import sys
import numpy as np

# Reference image path
ref_img_path = "data/me/Pic.jpg"

# Read reference image once and convert to RGB
ref_bgr = cv2.imread(ref_img_path)
if ref_bgr is None:
    print(f"Reference image not found at '{ref_img_path}'. Exiting.")
    sys.exit(1)
ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

# Pre-compute reference embedding once (much faster!)
print("Computing reference embedding...")
ref_embedding = DeepFace.represent(img_path=ref_rgb, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
print("Reference embedding computed!")

# ArcFace threshold (cosine distance)
threshold = 0.68

# OpenCV face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

def draw_status(frame, verified, distance=None, threshold_val=None):
    # Top-left box settings
    box_x, box_y = 10, 10
    box_w, box_h = 120, 80

    if verified:
        bg_color = (0, 200, 0)   # green background (BGR)
    else:
        bg_color = (0, 0, 200)   # red background (BGR)

    # Filled background rectangle
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), bg_color, -1)

    # White mark color and thickness
    mark_color = (255, 255, 255)
    thickness = 6

    if verified:
        # Draw a check mark: two lines forming the tick
        p1 = (box_x + 15, box_y + box_h // 2)
        p2 = (box_x + box_w // 2 - 5, box_y + box_h - 18)
        p3 = (box_x + box_w - 15, box_y + 18)
        cv2.line(frame, p1, p2, mark_color, thickness, cv2.LINE_AA)
        cv2.line(frame, p2, p3, mark_color, thickness, cv2.LINE_AA)
    else:
        # Draw a cross: two diagonal lines
        cv2.line(frame, (box_x + 15, box_y + 15), (box_x + box_w - 15, box_y + box_h - 15), mark_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (box_x + box_w - 15, box_y + 15), (box_x + 15, box_y + box_h - 15), mark_color, thickness, cv2.LINE_AA)

    # Display confidence score below the status box (only if we have valid data)
    if distance is not None and threshold_val is not None:
        # Calculate confidence score (lower distance = higher confidence)
        # Convert distance to a 0-100 confidence score
        confidence = max(0, min(100, (1 - (distance / threshold_val)) * 100))
        
        # Display distance and confidence
        text_y = box_y + box_h + 25
        cv2.putText(frame, f"Distance: {distance:.3f}", (box_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (box_x, text_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Threshold: {threshold_val:.3f}", (box_x, text_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # Display "No data" when distance is None
        text_y = box_y + box_h + 25
        cv2.putText(frame, "No verification data", (box_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2, cv2.LINE_AA)

# Optional: verify every N frames to reduce load (set to 1 to verify each frame)
frame_interval = 2
frame_count = 0
verified = False
distance = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using Haar cascade (for visualization & to decide when to verify)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    face_present = len(faces) > 0

    # By default, no rectangle drawn; if faces found, draw them
    if face_present:
        # Draw rectangles for all detected faces (color depends on verification)
        for (x, y, w, h) in faces:
            # Rectangle color: green if verified, yellow if face present but not verified
            if verified:
                rect_color = (0, 255, 0)
            else:
                rect_color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

    # Convert current frame to RGB (DeepFace expects RGB arrays)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run verification only every `frame_interval` frames (and only if a face detected)
    if frame_count % frame_interval == 0 and face_present:
        # Use the first detected face region as the candidate for verification
        x, y, w, h = faces[0]
        # Add small padding around the face crop
        pad = int(0.2 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        crop_rgb = frame_rgb[y1:y2, x1:x2]

        try:
            # Get embedding for current frame
            frame_embedding = DeepFace.represent(img_path=crop_rgb, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
            
            # Calculate cosine distance manually
            ref_arr = np.array(ref_embedding)
            frame_arr = np.array(frame_embedding)
            distance = 1 - np.dot(ref_arr, frame_arr) / (np.linalg.norm(ref_arr) * np.linalg.norm(frame_arr))
            
            # Verify based on threshold
            verified = distance < threshold
            
        except Exception as e:
            # If verification fails (e.g., face not clear), keep verified False
            verified = False
            distance = None
    elif frame_count % frame_interval == 0 and not face_present:
        # No face to verify
        verified = False
        distance = None

    frame_count += 1

    # Draw status indicator (tick/cross) in top-left corner with confidence score
    draw_status(frame, verified, distance, threshold)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()