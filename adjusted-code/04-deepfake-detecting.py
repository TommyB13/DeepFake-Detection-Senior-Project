import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
from tqdm import tqdm

# Load the trained model
model_path = './tmp_checkpoint/best_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
model = load_model(model_path, custom_objects={'FixedDropout': tf.keras.layers.Dropout})

# Initialize MTCNN face detector
face_detector = MTCNN(keep_all=True)

# Set input size
input_size = 64

# Create output folders for real and fake frames
output_base_path = './output'
real_output_path = os.path.join(output_base_path, 'real')
fake_output_path = os.path.join(output_base_path, 'fake')
os.makedirs(real_output_path, exist_ok=True)
os.makedirs(fake_output_path, exist_ok=True)

# Function to preprocess the face
def preprocess_face(face):
    face = face.resize((input_size, input_size), Image.LANCZOS)
    face = np.array(face).astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Function to analyze a video for DeepFake detection and save the frames
def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}. Please check the path and try again.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    deepfake_count = 0
    real_count = 0

    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert frame to PIL Image
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Detect faces in the frame
            faces, _ = face_detector.detect(frame_image)
            if faces is not None:
                for i, box in enumerate(faces):
                    x, y, width, height = [int(coord) for coord in box]
                    face_img = frame_image.crop((x, y, x + width, y + height))

                    # Preprocess the detected face
                    face_processed = preprocess_face(face_img)

                    # Predict using the trained model
                    prediction = model.predict(face_processed)[0][0]

                    # Threshold to classify as deepfake or real
                    if prediction > 0.5:
                        label = 'DeepFake'
                        deepfake_count += 1
                        output_folder = fake_output_path
                    else:
                        label = 'Real'
                        real_count += 1
                        output_folder = real_output_path

                    # Save the cropped face image to the corresponding folder
                    output_filename = f"frame{frame_count}_face{i}_{label}.png"
                    output_filepath = os.path.join(output_folder, output_filename)
                    face_img.save(output_filepath)

            pbar.update(1)

    cap.release()

    # Print final results
    print(f'Total Frames Analyzed: {frame_count}')
    print(f'Real Frames Detected: {real_count}')
    print(f'DeepFake Frames Detected: {deepfake_count}')

    if deepfake_count > real_count:
        print('The video is likely a DeepFake.')
    else:
        print('The video is likely Real.')

# Path to the video to be analyzed
video_path = '/home/tmbennett/senior-project/Real-Tom-Cruise-2.mp4'

# Run the DeepFake detection and save the frames
detect_deepfake(video_path)

