import cv2
import os
import math

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

def split_video_into_frames(video_path, base_output_folder):
    # Get the filename without extension to create a specific folder for this video
    video_name = get_filename_only(video_path)
    output_folder = os.path.join(base_output_folder, video_name)

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video from the specified path
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

    # Loop through each frame of the video
    while True:
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Current frame number
        ret, frame = cap.read()

        # If there are no more frames to read, exit the loop
        if not ret:
            break

        # Save frames at intervals equal to the frame rate (approximately 1 frame per second)
        if frame_id % math.floor(frame_rate) == 0:
            # Resize frame based on predefined conditions
            if frame.shape[1] < 300:
                scale_ratio = 2
            elif frame.shape[1] > 1900:
                scale_ratio = 0.33
            elif 1000 < frame.shape[1] <= 1900:
                scale_ratio = 0.5
            else:
                scale_ratio = 1

            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            dim = (width, height)
            new_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            # Generate the output path for the frame
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:03d}.png")

            # Save the current frame as an image file
            cv2.imwrite(frame_filename, new_frame)

            # Print the saved frame information
            print(f"Saved: {frame_filename}")

            frame_count += 1

    # Release the video capture object
    cap.release()
    print("Finished extracting frames.")

# Example usage
video_path = 'input_video.mp4'  # Replace with the path to your video file
base_output_folder = 'output_frames'  # Replace with the desired base output folder
split_video_into_frames(video_path, base_output_folder)

