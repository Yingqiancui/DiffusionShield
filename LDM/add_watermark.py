import os
import sys
import random
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse  
from diffusers import AutoencoderKL
import cv2

def process_images(block_width, image_size, dir_wmlist, dataset, save_dif, wm_patches_dir):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", revision=None)
    vae = vae.to('cuda')

    transform = transforms.Compose([
        transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=dataset, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    wmpatches = torch.load(wm_patches_dir)

    if type(dir_wmlist)==str:
        wmlist = torch.load(dir_wmlist)
    else: 
        wmlist= dir_wmlist
    count = int(image_size / block_width)

    m=0
    for data, _ in loader: 
        m+=1
        data2 = (data-0.5)*2
        print(data2.shape)
        latents = vae.encode(data2.to(torch.float32).to('cuda')).latent_dist.sample()
        for i in range(count):
            for j in range(count):
                index = count * i + j
                if wmlist[index]!=0:
                    latents[0][:,i*block_width:(i+1)*block_width, j*block_width:(j+1)*block_width] += wmpatches[0]
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        decoded_data=(image[0].detach().cpu())  
        img = np.array(decoded_data)

        img = np.transpose(img, (1,2,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        del decoded_data, image
        cv2.imwrite(f'{save_dif}/mylabel{m}.png', img * 255)

def main():
    parser = argparse.ArgumentParser(description="Process images with specified block width and image size.")
    parser.add_argument("--block_width", type=int, default=16, help="Width of each block.")
    parser.add_argument("--image_size", type=int, default=64, help="Size of the images.")
    parser.add_argument("--dir_wmlist", type=str, default=[0,1,0,1]*4, help="directory of the watermark string")
    parser.add_argument("--dataset", type=str, default='', help="Dataset: if not cifar10, indicate the directory")
    parser.add_argument("--save_dif", type=str, default='./watermarked_img/cf10/')
    parser.add_argument("--wmpatches_dir", type=str, default='./trained_patches/wm_patches.pt')
    
    args = parser.parse_args()
    if not os.path.exists(args.save_dif):
        os.makedirs(args.save_dif)
    process_images(args.block_width, args.image_size, args.dir_wmlist, args.dataset, args.save_dif, args.wmpatches_dir)

if __name__ == "__main__":
    main()
