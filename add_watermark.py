import os
import sys
import random
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse  

def process_images(block_width, image_size, dir_wmlist, dataset, save_dif, wm_patches_dir):

    transform = transforms.Compose([
        transforms.ToTensor()])

    if dataset=='cifar10':
        dataset = torchvision.datasets.CIFAR10('./cifar10_data', train=True, download=True)
    else:
        dataset = torchvision.datasets.ImageFolder(root=dataset)

    wmpatches = np.array(torch.load(wm_patches_dir))

    if type(dir_wmlist)==str:
        wmlist = torch.load(dir_wmlist)
    else: 
        wmlist= dir_wmlist

    count = int(image_size / block_width)

    for m, (img, label) in enumerate(dataset):
        # add watermark to the 'bird' class
        if label == 2:
            img = transform(img)  
            img = np.transpose(img.numpy(), (1, 2, 0)) 
            for i in range(count):
                for j in range(count):
                    index = count * i + j
                    if wmlist[index]!=0:
                        img[i*block_width:(i+1)*block_width, j*block_width:(j+1)*block_width] += wmpatches[wmlist[index]-1]

            img = np.clip(img, 0, 1)
            img = np.transpose(img, (2, 0, 1))  
            save_image(torch.tensor(img), f'{save_dif}/mylabel{label}_{m}.png')

def main():
    parser = argparse.ArgumentParser(description="Process images with specified block width and image size.")
    parser.add_argument("--block_width", type=int, default=4, help="Width of each block.")
    parser.add_argument("--image_size", type=int, default=32, help="Size of the images.")
    parser.add_argument("--dir_wmlist", type=str, default='example.pt', help="directory of the watermark string")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset: if not cifar10, indicate the directory")
    parser.add_argument("--save_dif", type=str, default='./watermarked_img/cf10/8_255')
    parser.add_argument("--wmpatches_dir", type=str, default='./trained_patches/wm_patches.pt')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dif):
        os.makedirs(args.save_dif)
        
    process_images(args.block_width, args.image_size, args.dir_wmlist, args.dataset, args.save_dif, args.wmpatches_dir)

if __name__ == "__main__":
    main()
