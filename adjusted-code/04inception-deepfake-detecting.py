import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# Load the trained InceptionV3 model
model_path = './tmp_checkpoint_inception/best_inception_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")

model = load_model(model_path)

# Initialize MTCNN face detector
face_detector = MTCNN(keep_all=True, device='cuda' if tf.config.list_physical_devices('GPU') else 'cpu')

# Set input size for InceptionV3
input_size = 299

# Create output folders for real and fake frames
output_base_path = './output'
real_output_path = os.path.join(output_base_path, 'real')
fake_output_path = os.path.join(output_base_path, 'fake')
os.makedirs(real_output_path, exist_ok=True)
os.makedirs(fake_output_path, exist_ok=True)

# Set log file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(output_base_path, f'detection_log_{timestamp}.txt')
log_file = open(log_file_path, 'w')

# Function to preprocess the face for InceptionV3
def preprocess_face(face):
    face = face.resize((input_size, input_size), Image.LANCZOS)
    face = np.array(face).astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Function to analyze a video for DeepFake detection
def detect_deepfake(video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}. Please check the path and try again.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    deepfake_count = 0
    real_count = 0
    face_count = 0

    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)} @ threshold {threshold}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = face_detector.detect(frame_image)

            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face_img = frame_image.crop((x1, y1, x2, y2))
                    face_processed = preprocess_face(face_img)
                    prediction = model.predict(face_processed, verbose=0)[0][0]
                    face_count += 1

                    if prediction > threshold:
                        label = 'DeepFake'
                        deepfake_count += 1
                        output_folder = fake_output_path
                    else:
                        label = 'Real'
                        real_count += 1
                        output_folder = real_output_path

                    output_filename = f"{os.path.basename(video_path).split('.')[0]}_frame{frame_count}_face{i}_{label}.png"
                    output_filepath = os.path.join(output_folder, output_filename)
                    face_img.save(output_filepath)

            pbar.update(1)

    cap.release()
    percent_deepfake = (deepfake_count / face_count) if face_count else 0
    conclusion = 'DEEPFAKE' if percent_deepfake >= 0.5 else 'REAL'

    result_summary = (
        f"\nVideo: {os.path.basename(video_path)}\n"
        f"Threshold: {threshold}\n"
        f"Total frames analyzed: {frame_count}\n"
        f"Total faces detected: {face_count}\n"
        f"Real frames detected: {real_count}\n"
        f"Fake frames detected: {deepfake_count}\n"
        f"Percent deepfake frames: {percent_deepfake:.2%}\n"
        f"Conclusion: {conclusion}\n"
    )

    print(result_summary)
    log_file.write(result_summary)
    log_file.flush()

# Video files to analyze
video_files = [
    'Biden-Cruise-Deepfake.mp4',
    'Deep-Tom-Cruise.mp4',
    'Biden2.mp4',
    'Real-Tom-Cruise.mp4',
    'Real-Tom-Cruise-2.mp4'
]

# Thresholds to evaluate
thresholds = [0.2, 0.5, 0.75]

# Run detection for each video and threshold
for video in video_files:
    video_path = os.path.join('../senior-project', video)
    for thresh in thresholds:
        detect_deepfake(video_path, thresh)

log_file.close()

