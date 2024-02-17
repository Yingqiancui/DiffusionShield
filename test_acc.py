import io
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from resnet import ResNet_modified
import argparse

class JpegCompressionTransform:
    def __init__(self, quality=20):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

class GaussianNoiseTransform:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def load_watermark(filename):
    if type(filename)==str:
        wm = torch.load(filename)
    else:
        wm = filename
    wmlist = []
    for bit in wm:
        wmlist.extend([bit // 2, bit % 2])
    return wmlist, wm

def load_dataset(root_path, transform):
    testset = datasets.ImageFolder(root=root_path, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    return testloader

def load_pretrained_model(checkpoint_path, device):
    net = ResNet_modified()
    checkpoint = torch.load(checkpoint_path)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    net.eval()
    return net

def evaluate_model(testloader, net, wmlist, wm, count, block_width, device):
    
    count_correct = 0
    count_total = 0

    count_correct2 = 0
    count_total2 = 0
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device).float()

            outputlist = []
            blocks = []
            for i in range(count):
                for j in range(count):
                    block = inputs[:, :, block_width*i:block_width*(i+1), block_width*j:block_width*(j+1)]
                    blocks.append(block)
            blocks = torch.cat(blocks, dim=0) 
            
            outputs = net(blocks)
            preds = torch.argmax(outputs, dim=1)
            
            for pred in preds:
                outputlist.extend([pred.item() // 2, pred.item() % 2])
            
            assert len(outputlist) == len(wmlist), "Output list and watermark list must be of the same length"
            count_correct += sum(o == w for o, w in zip(outputlist, wmlist))
            count_total += len(wmlist)
            count_correct2 += sum(o == w for o, w in zip(preds, wm))
            count_total2 += len(wm)

    accuracy = count_correct / count_total
    accuracy2 = count_correct2 / count_total2
    print(f'Bit Accuracy: {accuracy}; Classifier Accuracy: {accuracy2}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Watermark detection")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use for computation.')
    parser.add_argument('--block_width', type=int, default=4, help='Width of each block in the image.')
    parser.add_argument('--image_size', type=int, default=32, help='Size of the images.')
    parser.add_argument('--wm_filename', type=str or list, default=[0]*64, help='Path to the watermark string file.')
    parser.add_argument('--checkpoint_path', type=str, default='./trained_model/classifier.pt', help='Path to the trained model checkpoint.')
    parser.add_argument('--data_root', type=str, default='./watermarked_img/cf10/',help='Root directory of the watermarked images.')
    parser.add_argument('--attack_type', type=str, default='None',help='Choose from "None", "grey", "blur", "noise", and "jpeg"')
    args = parser.parse_args()

    wmlist, wm = load_watermark(args.wm_filename)

    if args.attack_type == 'grey':
        transform_train = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
        ])
    elif args.attack_type == 'blur':
        transform_train = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=5, sigma=(1)),
        ])
    elif args.attack_type == 'jpeg':
        transform_train = transforms.Compose([
            JpegCompressionTransform(quality=80),
            transforms.ToTensor(),
        ])
    elif args.attack_type == 'noise':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            GaussianNoiseTransform(mean=0, std=0.1),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    testloader = load_dataset(args.data_root, transform_train)
    net = load_pretrained_model(args.checkpoint_path, args.device)

    count = int(args.image_size / args.block_width)
    evaluate_model(testloader, net, wmlist, wm, count, args.block_width, args.device)
