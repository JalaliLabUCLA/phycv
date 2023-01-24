"""
Run VEVID on a video. 
All steps (load_img -> init_kernel -> apply_kernel) are performed seperately.
For fixed kernel, we only need to initialize the kernel once for the first frame 

"""

import os

import imageio.v3 as iio
import numpy as np
import torch
import torchvision

from phycv import VEVID_GPU


def main():
    # indicate the video to be processed
    vid = torchvision.io.read_video("./assets/input_videos/video_building.mp4")
    print("video loaded!")

    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get how many frames are in the video
    # create a empty array to store the VEViD output
    vid_frames = vid[0]
    length = vid_frames.shape[0]
    frame_h = vid_frames[0].shape[0]
    frame_w = vid_frames[0].shape[1]
    vevid_out_vid = np.zeros((length, 3, frame_h, frame_w))

    # indicate VEViD-lite parameters
    b = 0.5
    G = 0.6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run VEViD for each frame
    vevid = VEVID_GPU(device=device)

    for i in range(length):
        frame = torch.permute(vid_frames[i], (2, 0, 1)) / 255.0
        frame = frame.to(device)
        vevid.load_img(img_array=frame)
        vevid.apply_kernel(b, G, lite=True)
        vevid_out_vid[i] = vevid.vevid_output.cpu().numpy()

    print("create video...")
    # save the results for each frame
    concat_frames = []
    for i in range(length):
        raw_frame = vid_frames[i].numpy()
        vevid_frame = (np.transpose(vevid_out_vid[i], (1, 2, 0)) * 255).astype(np.uint8)
        concat_frame = np.concatenate((raw_frame, vevid_frame), 1)
        concat_frames.append(concat_frame)

    # create video from the processed frames
    iio.imwrite("output/VEViD_lite_video_demo.mp4", concat_frames, fps=20)


if __name__ == "__main__":
    main()
