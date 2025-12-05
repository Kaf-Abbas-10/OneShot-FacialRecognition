from deepface import DeepFace
import cv2

# Reference image
ref_img = "data/me/Pic.jpg"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save current frame to temp file
    cv2.imwrite("current.jpg", frame)
    
    try:
        result = DeepFace.verify(img1_path=ref_img, img2_path="current.jpg", model_name="ArcFace")
        if result["verified"]:
            print("You are in the frame!")
        else:
            print("Not you")
    except:
        pass  # No face detected
    
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
