import cv2
import mediapipe as mp
import os
import csv

# -----------------------------
# Setup: Folders & Mediapipe
# -----------------------------

# Input folder where your images are stored
INPUT_FOLDER = "input"

# Output folders to save results
OUTPUT_FOLDER = "output"
ANOMALY_FOLDER = "anomaly"

# Create output folders if they don't exist
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs(ANOMALY_FOLDER, exist_ok=True)

# CSV file to log results
CSV_FILE = "results.csv"

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


# -----------------------------
# Function: Process one image
# -----------------------------
def process_image(img_path):
    """Reads an image, detects all faces, and saves the largest one as subject."""
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return None, None, None

    # Convert image to RGB (Mediapipe requirement)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb)

    if not results.detections:
        print(f"[INFO] No faces found in {img_path}")
        return os.path.basename(img_path), None, 0

    # Store bounding boxes of detected faces
    faces = []
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x1 = int(bboxC.xmin * w)
        y1 = int(bboxC.ymin * h)
        x2 = int((bboxC.xmin + bboxC.width) * w)
        y2 = int((bboxC.ymin + bboxC.height) * h)
        faces.append((x1, y1, x2, y2))

    # Pick the largest face (closest to camera)
    largest_face = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    # Save subject face (largest one)
    subject_face = image[largest_face[1]:largest_face[3], largest_face[0]:largest_face[2]]
    subject_path = os.path.join(OUTPUT_FOLDER, "subject_" + os.path.basename(img_path))
    cv2.imwrite(subject_path, subject_face)

    # Save anomalies (all other faces)
    anomaly_count = 0
    for i, (x1, y1, x2, y2) in enumerate(faces):
        if (x1, y1, x2, y2) != largest_face:
            anomaly_face = image[y1:y2, x1:x2]
            anomaly_path = os.path.join(ANOMALY_FOLDER, f"anomaly_{i}_" + os.path.basename(img_path))
            cv2.imwrite(anomaly_path, anomaly_face)
            anomaly_count += 1

    return os.path.basename(img_path), subject_path, anomaly_count


# -----------------------------
# Main Program
# -----------------------------
def main():
    # Open CSV for logging
    with open(CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Original Image", "Subject Saved", "Anomaly Count"])

        # Process every image in input folder
        for filename in os.listdir(INPUT_FOLDER):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(INPUT_FOLDER, filename)
                original, subject, anomalies = process_image(img_path)

                if original is not None:
                    writer.writerow([original, subject, anomalies])
                    print(f"[INFO] {original}: {anomalies+1} faces detected ✅ "
                          f"(Subject saved, {anomalies} anomalies)")


if __name__ == "__main__":
    main()
    