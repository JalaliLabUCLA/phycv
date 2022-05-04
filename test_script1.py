'''
test script 1: All steps (load_img -> init_kernel -> apply_kernel -> create_edge) are performed in a single run method. 
               This is for users who want to get the result on a single image with indicated PST/PAGE parameters. 

test script 2: Each step of PST/PAGE is performed seperately, load_img -> init_kernel -> apply_kernel -> create_edge. 
               This is for video processing where different frames need to be loaded but the same kernel applies to all the frames 
'''

# import
from phycv import PAGE, PAGE_GPU, PST, PST_GPU
import torch
import matplotlib.pyplot as plt

# indicate image file, height and width of the image, and GPU device (if applicable)
img_file = './input_images/jet_engine.jpeg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# run PST CPU version
pst = PST()
pst_output = pst.run(img_file=img_file, phase_strength=0.3, warp_strength=12, sigma_LPF=0.1, thresh_min=-1, thresh_max=0.0019, morph_flag=1)
figure1 = plt.figure(1)
plt.title('PST CPU')
plt.imshow(pst_output, cmap='gray')

# run PST GPU version
pst = PST_GPU(device=device)
pst_output = pst.run(img_file=img_file, phase_strength=0.3, warp_strength=12, sigma_LPF=0.1, thresh_min=-1, thresh_max=0.0019, morph_flag=1)
figure2 = plt.figure(2)
plt.title('PST GPU')
plt.imshow(pst_output.cpu().numpy(), cmap='gray')

# run PAGE CPU version
page = PAGE(direction_bins=10)
page_edge = page.run(img_file=img_file, mu_1=0, mu_2=0.35, sigma_1=0.08, sigma_2=0.7, S1=0.3, S2=0.3, sigma_LPF=0.1, thresh_min=-1, thresh_max=0.0003, morph_flag=1)
figure3 = plt.figure(3)
plt.title('PAGE CPU')
plt.imshow(page_edge)

# run PAGE GPU version
page = PAGE_GPU(direction_bins=10, device=device)
page_edge = page.run(img_file=img_file, mu_1=0, mu_2=0.35, sigma_1=0.08, sigma_2=0.7, S1=0.3, S2=0.3, sigma_LPF=0.1, thresh_min=-1, thresh_max=0.0003, morph_flag=1)
figure4 = plt.figure(4)
plt.title('PAGE GPU')
plt.imshow(page_edge.cpu().numpy())

plt.show()
