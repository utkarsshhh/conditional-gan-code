import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

class Generator(nn.Module):
    def __init__(self,z_dim = 10,im_chan = 3, hidden_dim=64):
        super(self,Generator).__init__()
        self.z_dim = z_dim

        self.gen = nn.Sequential(
            self.make_gen_block(z_dim,hidden_dim*8),
            self.make_gen_block(hidden_dim*8,hidden_dim*4),
            self.make_gen_block(hidden_dim*4,hidden_dim*2),
            self.make_gen_block(hidden_dim*2,hidden_dim),
            self.make_gen_block(hidden_dim,im_chan,kernel_size = 4,final_layer = True)
        )

    def make_gen_block(self,input_channels,output_channels,kernel_size = 3,stride = 2,final_layer = False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels= input_channels,out_channels=output_channels,kernel_size=kernel_size,stride = stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels= input_channels,out_channels=output_channels,kernel_size=kernel_size,stride = stride),
                nn.Tanh()
            )

    def forward(self,noise):
        x = noise.view(len(noise),self.z_dim,1,1)
        return self.gen(x)


def generate_noise(n_samples,z_dim,device = 'cpu'):
    return torch.randn(n_samples,z_dim,device=device)


class Classifier(nn.Module):
    def __init__(self,im_chan=3,n_classes= 2,hidden_dim=64):
        super(self,Classifier).__init__()
        self.classifier = nn.Sequential(
            self.make_classifier_block(im_chan,hidden_dim),
            self.make_classifier_block(hidden_dim,hidden_dim*2),
            self.make_classifier_block(hidden_dim * 2, hidden_dim * 4, stride=3),
            self.make_classifier_block(hidden_dim * 4, n_classes, final_layer=True)
        )

    def make_classifier_block(self,input_channels,output_channels,kernel_size = 4,stride =2,final_layer = False ):

        if (final_layer):
            return nn.Conv2d(input_channels,out_channels=output_channels,kernel_size=kernel_size,stride=stride)
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=input_channels,out_channels= output_channels,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2,inplace=True)
            )

    def forwrad(self,image):
        class_pred = self.classifier(image)
        return class_pred.view(len(class_pred),-1)

z_dim = 64
batch_size = 128
device = 'cuda'