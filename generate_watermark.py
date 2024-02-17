''' 
Portions of this file are derived from DiffJPEG by Marcela Lomnitz
GitHub: https://github.com/mlomnitz/DiffJPEG
Licensed under MIT License
'''

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
from DiffJPEG.DiffJPEG import DiffJPEG

class generate_watermark(Base):
    def __init__(self, model,
                device,
                lr_train=0.002,
                epoch_num=100,
                epsilon = 8 / 2550., 
                clip_max= 8/255,
                clip_min = -8/255, 
                patch_size = 4,
                image_size = 32,
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
        self.clip_max=clip_max
        self.clip_min=clip_min
        self.patch_size=patch_size
        self.image_size=image_size
        self.B=B
        self.count=int(image_size/patch_size)
        self.augment_type=augment_type
        self.jpeg = DiffJPEG(height=32, width=32, differentiable=True, quality=80)
        self.jpeg.to(self.device)
        self.model_save_path = model_save_path
        self.patches_save_path = patches_save_path
  
    def gaussian_noise(self, x, severity):
        c = torch.tensor([0.04, 0.06, .08, .09, .10][severity - 1], device=self.device)
        noise = torch.normal(0, c, size=x.shape, device=self.device)
        return (x.to(self.device) + noise).clip(0, 1)
    
    def generate_wm(self, train_loader, test_loader):

        initial = list(torch.rand(3,self.patch_size,self.patch_size).unsqueeze(dim=0))
        wm_patches=(self.B-1)*list(initial)
        torch.manual_seed(100)

        best_acc=self.test(wm_patches,test_loader, train=False)
        best_acc2=self.test(wm_patches,train_loader, train=True)
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
                
                wm_patches_save = [np.transpose(per.cpu().detach().numpy(), (1, 2, 0)) for per in wm_patches]
                torch.save(wm_patches_save, self.patches_save_path)

    def test(self, wm_patches,test_loader,train=False):

        self.model.eval()
        test_loss = 0
        correct = 0

        for data, _ in test_loader:
            randomlist=[]
            for _ in range(0,self.count**2):
                n = random.randint(0,3)
                randomlist.append(n)
            for m in range(data.shape[0]):  
                for i in range(0,self.count):
                    for j in range(0,self.count):
                        if randomlist[self.count*i+j]!=0:
                            data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]= \
                                (data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]+ \
                                 wm_patches[randomlist[self.count*i+j]-1].cpu().detach()).clamp(0,1)

            if self.augment_type=='grey':
                grey=transforms.Grayscale(num_output_channels=3)
                data=grey(data)
            elif self.augment_type=='blur':
                blur=transforms.GaussianBlur(kernel_size=5, sigma=(1))
                data=blur(data)
            elif self.augment_type=='noise':
                data = self.gaussian_noise(data, severity=5)
            elif self.augment_type=='jpeg':
                data=self.jpeg(data.to(self.device))
                
            datalist=[] 
            target_list = randomlist*data.shape[0]
            for m in range(data.shape[0]):           
                for i in range(0,self.count):
                    for j in range(0,self.count):
                        content=data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]
                        content=content[None,:,:,:]
                        datalist.append(content)
            
            train_data = torch.concatenate(datalist, dim=0).to(self.device)
            train_target =torch.tensor(target_list).to(self.device)
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
            n = random.randint(0,3)
            randomlist.append(n)
        
        for m in range(data.shape[0]):  
            for i in range(0,self.count):
                for j in range(0,self.count):
                    if randomlist[self.count*i+j]!=0:
                        data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]= \
                            (data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size] \
                             +wm_patches[randomlist[self.count*i+j]-1].cpu().detach()).clamp(0,1)

        if self.augment_type=='grey':
            grey=transforms.Grayscale(num_output_channels=3)
            data=grey(data)
        elif self.augment_type=='blur':
            blur=transforms.GaussianBlur(kernel_size=5, sigma=(1))
            data=blur(data)
        elif self.augment_type=='noise':
            data = self.gaussian_noise(data, severity=5)
        elif self.augment_type=='jpeg':
            data=self.jpeg(data.to(self.device))
              
        datalist=[] 
        target_list = randomlist*data.shape[0]
        for m in range(data.shape[0]):           
            for i in range(0,self.count):
                for j in range(0,self.count):
                    content=data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]
                    content=content[None,:,:,:]
                    datalist.append(content)
        
        train_data = torch.concatenate(datalist, dim=0).to(self.device)
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
            n = random.randint(0,3)
            randomlist.append(n)
        for m in range(data.shape[0]):  
            for i in range(0,self.count):
                for j in range(0,self.count):
                    if randomlist[self.count*i+j]!=0:
                        data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]= \
                            (data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]+ \
                            wm_patches[randomlist[self.count*i+j]-1]).clamp(0,1)
        if self.augment_type=='grey':
            grey=transforms.Grayscale(num_output_channels=3)
            data=grey(data)
        elif self.augment_type=='blur':
            blur=transforms.GaussianBlur(kernel_size=5, sigma=(1))
            data=blur(data)
        elif self.augment_type=='noise':
            data = self.gaussian_noise(data, severity=5)
        elif self.augment_type=='jpeg':
            data=self.jpeg(data.to(self.device))

        datalist=[] 
        target_list = randomlist*data.shape[0]
        for m in range(data.shape[0]):           
            for i in range(0,self.count):
                for j in range(0,self.count):
                    content=data[m,:,i*self.patch_size:i*self.patch_size+self.patch_size,j*self.patch_size:j*self.patch_size+self.patch_size]
                    content=content[None,:,:,:]
                    datalist.append(content)
       
        train_data = torch.concatenate(datalist, dim=0).to(self.device)
        train_target =torch.tensor(target_list).to(self.device)

        opt = optim.SGD(wm_patches, lr=1e-3)
        opt.zero_grad()
      
        loss = nn.CrossEntropyLoss()(self.model(train_data), train_target)
        loss.backward()
            
        for m in range(len(wm_patches)):
            wm_patches[m] = wm_patches[m] - self.epsilon * wm_patches[m].grad.data.sign()
            wm_patches[m] = torch.clamp(wm_patches[m],self.clip_min, self.clip_max)
            wm_patches[m] =  wm_patches[m].detach()

        opt.zero_grad()

        return wm_patches
    

