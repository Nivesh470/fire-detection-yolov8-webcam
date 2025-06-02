from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained fire/smoke detection model
model = YOLO("best (1).pt")  # Ensure 'best (1).pt' is in your project directory

# Open webcam
cap = cv2.VideoCapture(0)

# Parameters for flame detection (adjust as needed)
min_flame_area = 30  # Minimum area (in pixels) to consider a flame
flicker_threshold = 10  # Minimum change in mask pixels between frames to consider flicker
previous_mask = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # --- Image Preprocessing for Small Flame Detection ---
    # 1. Apply Gaussian Blur (reduce noise, but keep it mild)
    blurred_frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Smaller kernel

    # 2. Convert to HSV color space
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # 3. Define a more specific range for lighter flame colors (often more yellow/orange)
    lower_flame = np.array([10, 100, 100])  # Adjust these values carefully
    upper_flame = np.array([40, 255, 255])
    flame_mask = cv2.inRange(hsv_frame, lower_flame, upper_flame)

    # 4. Apply morphological operations (even milder)
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(flame_mask, kernel, iterations=1)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)

    # --- Flicker Detection (Experimental) ---
    is_flickering = False
    if previous_mask is not None:
        diff_mask = cv2.absdiff(eroded_mask, previous_mask)
        flicker_pixels = np.sum(diff_mask > 0)
        if flicker_pixels > flicker_threshold:
            is_flickering = True
    previous_mask = eroded_mask.copy()

    # --- Run YOLOv8 model on the original frame ---
    results = model(frame)

    # Draw detection results
    annotated_frame = results[0].plot()

    # --- Combine Mask and YOLOv8 Detection ---
    detected_flame_by_mask = False
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_flame_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle for mask detection
            cv2.putText(annotated_frame, "Potential Flame (Mask)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            detected_flame_by_mask = True
            break  # Consider only the largest potential flame region

    # --- Enhanced Logic for Fire Detection Reporting ---
    detected_by_yolo = False
    for det in results[0].boxes:
        confidence = det.conf[0]
        if confidence > 0.5:  # Adjust confidence threshold if needed
            detected_by_yolo = True
            break

    if detected_by_yolo or (detected_flame_by_mask and is_flickering):
        cv2.putText(annotated_frame, "FIRE DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with detection overlays
    cv2.imshow("Fire Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
