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
pst.load_img(img_file=img_file)
pst.init_kernel(phase_strength=0.3, warp_strength=15)
pst.apply_kernel(sigma_LPF=0.15, thresh_min=-0.5, thresh_max=0.003, morph_flag=1)
figure1 = plt.figure(1)
plt.title('PST CPU')
plt.imshow(pst.pst_output, cmap='gray')

# run PST GPU version
pst = PST_GPU(device=device)
pst.load_img(img_file=img_file)
pst.init_kernel(phase_strength=0.3, warp_strength=15)
pst.apply_kernel(sigma_LPF=0.15, thresh_min=-0.5, thresh_max=0.003, morph_flag=1)
figure2 = plt.figure(2)
plt.title('PST GPU')
plt.imshow(pst.pst_output.cpu().numpy(), cmap='gray')

# run PAGE CPU version
page = PAGE(direction_bins=10)
page.load_img(img_file=img_file)
page.init_kernel(mu_1=0, mu_2=0.35, sigma_1=0.05, sigma_2=0.7, S1=0.8, S2=0.8)
page.apply_kernel(sigma_LPF=0.08, thresh_min=-1, thresh_max=0.0004, morph_flag=1)
page.create_page_edge()
figure3 = plt.figure(3)
plt.title('PAGE CPU')
plt.imshow(page.page_edge)

# run PAGE GPU version
page = PAGE_GPU(direction_bins=10, device=device)
page.load_img(img_file=img_file)
page.init_kernel(mu_1=0, mu_2=0.35, sigma_1=0.05, sigma_2=0.7, S1=0.8, S2=0.8)
page.apply_kernel(sigma_LPF=0.08, thresh_min=-1, thresh_max=0.0004, morph_flag=1)
page.create_page_edge()
figure4 = plt.figure(4)
plt.title('PAGE GPU')
plt.imshow(page.page_edge.cpu().numpy())

plt.show()
