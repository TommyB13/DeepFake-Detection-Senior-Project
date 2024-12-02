import cv2
import os
import torch
from facenet_pytorch import MTCNN

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

def extract_faces_from_frames_folder(frames_folder, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize MTCNN for face detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    # Loop through each frame in the folder
    for frame_filename in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame_filename)

        # Read the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Could not read frame {frame_filename}.")
            continue

        # Convert frame to RGB (MTCNN expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(rgb_frame)

        # If no faces are detected, skip this frame
        if boxes is None:
            continue

        # Crop and save each detected face with a margin
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]

            # Add margin to the bounding box
            margin_x = (x2 - x1) * 0.3  # 30% as the margin
            margin_y = (y2 - y1) * 0.3  # 30% as the margin

            x1 = int(x1 - margin_x)
            if x1 < 0:
                x1 = 0
            x2 = int(x2 + margin_x)
            if x2 > frame.shape[1]:
                x2 = frame.shape[1]
            y1 = int(y1 - margin_y)
            if y1 < 0:
                y1 = 0
            y2 = int(y2 + margin_y)
            if y2 > frame.shape[0]:
                y2 = frame.shape[0]

            # Crop the face with the margin
            cropped_face = frame[y1:y2, x1:x2]

            # Generate the output path for the cropped face
            face_filename = os.path.join(output_folder, f"{get_filename_only(frame_filename)}_face_{i:02d}.png")

            # Save the cropped face as an image file
            cv2.imwrite(face_filename, cropped_face)

            # Print the saved face information
            print(f"Saved: {face_filename}")

    print("Finished extracting faces from frames.")

# Example usage
frames_folder = 'output_frames/input_video'  # Replace with the path to your frames folder
output_folder = 'output_faces'  # Replace with the desired output folder for cropped faces
extract_faces_from_frames_folder(frames_folder, output_folder)
