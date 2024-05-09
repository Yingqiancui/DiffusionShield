import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
from PIL import Image as PILImage
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from base import Base
from io import BytesIO
from diffusers import AutoencoderKL

class generate_watermark(Base):
    def __init__(self, model,
                device,
                lr_train=0.002,
                epoch_num=100,
                epsilon = 0.01, 
                patch_size = 4,
                image_size = 32,
                budget = 0.007,
                augment_type='None',
                B=4,
                patches_save_path='',
                model_save_path=''
                ):
        
        if not torch.cuda.is_available():
            print('CUDA not availiable, using cpu...')
            self.device = 'cpu'
        else:
            self.device = device

        self.model = torch.nn.DataParallel(model.cuda())
        self.lr_train = lr_train
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_train, weight_decay=5e-4)
        self.device='cuda'
        self.epsilon=epsilon
        self.epoch_num=epoch_num
        self.patch_size=patch_size
        self.image_size=image_size
        self.B=B
        self.count=int(image_size/patch_size)
        self.augment_type=augment_type
        self.model_save_path = model_save_path
        self.patches_save_path = patches_save_path
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", revision=None)
        self.vae.requires_grad_(False)
        self.vae.to('cuda', dtype=torch.float32)
        self.budget = budget

    def gaussian_noise(self, x, severity):
        c = torch.tensor([0.04, 0.06, .08, .09, .10][severity - 1], device=self.device)
        noise = torch.normal(0, c, size=x.shape, device=self.device)
        return (x.to(self.device) + noise).clip(0, 1)
    
    def generate_wm(self, train_loader, test_loader):

        initial = list(torch.rand(4,self.patch_size,self.patch_size).unsqueeze(dim=0).to(self.device))
        wm_patches=(self.B-1)*list(initial)
        torch.manual_seed(100)

        best_acc=self.test(wm_patches,test_loader, train=False)
        for epoch in range(1, self.epoch_num + 1):

            print(epoch, flush = True)
            wm_patches=self.train_wm(wm_patches, train_loader)
            
            correct=self.test(wm_patches,test_loader, train=True)
            _=self.test(wm_patches,train_loader, train=False)

            if correct>=best_acc:
                print('saving ....')
                best_acc=correct

                state = {'net': self.model.state_dict(),}
                torch.save(state, self.model_save_path)
                
                wm_patches_save = [per for per in wm_patches]
                torch.save(wm_patches_save, self.patches_save_path)

    def test(self, wm_patches,test_loader,train=False):

        self.model.eval()
        test_loss = 0
        correct = 0

        for data, _ in test_loader:
            randomlist=[]
            for _ in range(0,self.count**2):
                n = random.randint(0,1)
                randomlist.append(n)
            data2 = (data-0.5)*2
            print(data.shape)
            latents = self.vae.encode(data2.to(torch.float32).to(self.device)).latent_dist.sample()

            for m in range(latents.shape[0]):  
                for i in range(0,self.count):
                    for j in range(0,self.count):
                        if randomlist[self.count*i+j]!=0:
                            latents[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]= \
                                (latents[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]+ \
                                 wm_patches[randomlist[self.count*i+j]-1].detach())
                            
            image = self.vae.decode(latents).sample
            decoded_data = (image / 2 + 0.5).clamp(0, 1)

            if self.augment_type=='grey':
                grey=transforms.Grayscale(num_output_channels=3)
                decoded_data=grey(decoded_data)
            elif self.augment_type=='blur':
                blur=transforms.GaussianBlur(kernel_size=5, sigma=(1))
                decoded_data=blur(decoded_data)
            elif self.augment_type=='noise':
                decoded_data = self.gaussian_noise(decoded_data, severity=5)
                
            datalist=[] 
            target_list = randomlist*decoded_data.shape[0]
            for m in range(decoded_data.shape[0]):           
                for i in range(0,self.count):
                    for j in range(0,self.count):
                        content=decoded_data[m,:,i*self.patch_size*8:i*self.patch_size*8+self.patch_size*8,j*self.patch_size*8:j*self.patch_size*8+self.patch_size*8]
                        content=content[None,:,:,:]
                        datalist.append(content)
            
            train_data = torch.concatenate(datalist, dim=0)
            train_target =torch.tensor(target_list).to(self.device)
            print(train_data.shape)
            output = self.model(train_data)
            test_loss += F.cross_entropy(output, train_target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
            correct += pred.eq(train_target.view_as(pred)).sum().item()

        test_loss /= (len(test_loader.dataset)*self.count**2)
       
        print('\n{} Loss: {:.5f}, Accuracy: {}/{} ({:.5f}%)\n'.format('Train' if train==True else 'Test',
            test_loss, correct, len(test_loader.dataset)*self.count**2,
            100. * correct / (len(test_loader.dataset)*self.count**2)))
        return correct


    def calculate_loss(self, output, target, redmode = 'mean'):
        """
        Calculate loss for training.
        """

        loss = F.cross_entropy(output, target, reduction = redmode)
        return loss
    
    def train_model(self,wm_patches,data,target):

        self.model.train()
        randomlist=[]
        for i in range(0,self.count**2):
            n = random.randint(0,1)
            randomlist.append(n)

        data2 = (data-0.5)*2
        latents = self.vae.encode(data2.to(torch.float32).to(self.device)).latent_dist.sample()

        for m in range(latents.shape[0]):  
            for i in range(0,self.count):
                for j in range(0,self.count):
                    if randomlist[self.count*i+j]!=0:
                        latents[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]= \
                            (latents[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size] \
                             +wm_patches[randomlist[self.count*i+j]-1].detach())
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        decoded_data=(image)

        if self.augment_type=='grey':
            grey=transforms.Grayscale(num_output_channels=3)
            decoded_data=grey(decoded_data)
        elif self.augment_type=='blur':
            blur=transforms.GaussianBlur(kernel_size=5, sigma=(1))
            decoded_data=blur(decoded_data)
        elif self.augment_type=='noise':
            decoded_data = self.gaussian_noise(decoded_data, severity=5)
              
        datalist=[] 
        target_list = randomlist*decoded_data.shape[0]
        for m in range(decoded_data.shape[0]):           
            for i in range(0,self.count):
                for j in range(0,self.count):
                    content=decoded_data[m,:,i*self.patch_size*8:i*self.patch_size*8+self.patch_size*8,j*self.patch_size*8:j*self.patch_size*8+self.patch_size*8]
                    content=content[None,:,:,:]
                    datalist.append(content)
        
        train_data = torch.concatenate(datalist, dim=0)
        train_target =torch.tensor(target_list).to(self.device)

        self.optimizer.zero_grad()
        
        output = self.model(train_data)
        loss = self.calculate_loss(output, train_target)

        loss.backward()
        self.optimizer.step()


    def train_wm(self, wm_patches, train_loader):

        wm_patches = [torch.tensor(imageArray.cpu().detach().numpy()).to(self.device) for imageArray in wm_patches]

        for _, (data, target) in enumerate(train_loader):
            self.train_model(wm_patches, data, target)
            for _ in range(10):
                wm_patches=self.update_wm(wm_patches, data ,target)
        return wm_patches
        

    def update_wm(self, wm_patches, data, target):
        self.model.eval()

        wm_patches = [per.requires_grad_() for per in wm_patches]
        data, target = data.to(self.device), target.to(self.device)
        randomlist=[]
        for _ in range(0,self.count**2):
            n = random.randint(0,1)
            randomlist.append(n)

        data2 = (data-0.5)*2
        latents = self.vae.encode(data2.to(torch.float32)).latent_dist.sample()

        for m in range(latents.shape[0]):  
            for i in range(0,self.count):
                for j in range(0,self.count):
                    if randomlist[self.count*i+j]!=0:
                        latents[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]= \
                            (latents[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]+ \
                            wm_patches[randomlist[self.count*i+j]-1])

        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        decoded_data=(image)

        if self.augment_type=='grey':
            grey=transforms.Grayscale(num_output_channels=3)
            decoded_data=grey(decoded_data)
        elif self.augment_type=='blur':
            blur=transforms.GaussianBlur(kernel_size=5, sigma=(1))
            decoded_data=blur(decoded_data)
        elif self.augment_type=='noise':
            decoded_data = self.gaussian_noise(decoded_data, severity=5)

        datalist=[] 
        diff=0
        target_list = randomlist*data.shape[0]
        for m in range(decoded_data.shape[0]):           
            for i in range(0,self.count):
                for j in range(0,self.count):
                    content=decoded_data[m,:,i*self.patch_size*8:i*self.patch_size*8+self.patch_size*8,j*self.patch_size*8:j*self.patch_size*8+self.patch_size*8]
                    diff+=torch.norm((decoded_data[m,:,i*self.patch_size*8:i*self.patch_size*8+self.patch_size*8,j*self.patch_size*8:j*self.patch_size*8+self.patch_size*8]\
                                      -data[m,:,i*self.patch_size*8:i*self.patch_size*8+self.patch_size*8,j*self.patch_size*8:j*self.patch_size*8+self.patch_size*8]),p=2)
                    content=content[None,:,:,:]
                    datalist.append(content)
       
        train_data = torch.concatenate(datalist, dim=0).to(self.device)
        train_target =torch.tensor(target_list).to(self.device)

        opt = optim.SGD(wm_patches, lr=1e-3)
        opt.zero_grad()
      
        loss = nn.CrossEntropyLoss()(self.model(train_data), train_target)+ self.budget*diff/16
        loss.backward()
            
        for m in range(len(wm_patches)):
            wm_patches[m] = wm_patches[m] - self.epsilon * wm_patches[m].grad.data.sign()
            wm_patches[m] =  wm_patches[m].detach()

        opt.zero_grad()

        return wm_patches
    

