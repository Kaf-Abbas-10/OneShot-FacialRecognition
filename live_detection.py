from deepface import DeepFace
import cv2
import sys

# Reference image path
ref_img_path = "data/me/Pic.jpg"

# Read reference image once and convert to RGB
ref_bgr = cv2.imread(ref_img_path)
if ref_bgr is None:
    print(f"Reference image not found at '{ref_img_path}'. Exiting.")
    sys.exit(1)
ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

# Build ArcFace model once (reduces per-frame overhead)
model = DeepFace.build_model("ArcFace")

cap = cv2.VideoCapture(0)

def draw_status(frame, verified):
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

# Optional: verify every N frames to reduce load (set to 1 to verify each frame)
frame_interval = 2
frame_count = 0
verified = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to RGB (DeepFace expects RGB arrays)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run verification only every `frame_interval` frames
    if frame_count % frame_interval == 0:
        try:
            result = DeepFace.verify(img1_path=ref_rgb, img2_path=frame_rgb, model_name="ArcFace", model=model)
            verified = bool(result.get("verified", False))
        except Exception:
            # If verification fails (e.g., no face detected), keep verified False
            verified = False

    frame_count += 1

    # Draw status indicator (tick/cross) in top-left corner
    draw_status(frame, verified)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
