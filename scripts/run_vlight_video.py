import os
import cv2
import numpy as np
import imageio.v3 as iio
from phycv import VLight

def main():
    # indicate the video to be processed
    video_file = "./assets/input_videos/video_building.mp4"
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    output_path = "./assets/output_results/video_building_vlight.png"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get video properties
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create an empty array to store the VLight output
    vlight_out_vid = np.zeros((length, frame_h, frame_w, 3), dtype=np.uint8)

    # VLight parameters
    v = 0.70
    vlight = VLight()

    # Process each frame
    concat_frames = []
    for i in range(length):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # VLight expects BGR frames, no need for conversion as OpenCV loads in BGR
        vlight_output = vlight.run_img_array(img_array=frame, v=v, color=False, lut=True)

        # Save the output into the array
        vlight_out_vid[i] = vlight_output

        # Concatenate original and VLight output side by side for comparison
        concat_frame = np.concatenate((frame, vlight_output), axis=1)
        concat_frames.append(concat_frame)

    print("Creating video...")

    # Create video from the processed frames
    iio.imwrite(os.path.join(output_path, "VLight_video_demo.mp4"), concat_frames, fps=fps)

    # Release video capture
    cap.release()
    print("Processing complete.")

if __name__ == "__main__":
    main()
