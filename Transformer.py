import torch 
import numpy 
import torchvision
print(torch.__version__)
print(torchvision.__version__)
from torchvision import datasets, transforms 
from torch.utils.data import Dataset, Subset
import sys
import os
import torchvision.transforms.functional as F

class DeterministicRandomRotation:
    def __init__(self, degrees, generator):
        self.degrees = degrees
        self.generator = generator

    def __call__(self, img):
        angle = torch.empty(1, generator=self.generator).uniform_(-self.degrees, self.degrees).item()
        return F.rotate(img, angle)


class DeterministicRandomCrop:
    def __init__(self, size, padding, generator):
        self.size = size
        self.padding = padding
        self.generator = generator

    def __call__(self, img):
        if self.padding:
            img = F.pad(img, self.padding)
        w, h = F.get_image_size(img)
        th, tw = self.size
        i = torch.randint(0, h - th + 1, size=(1,), generator=self.generator).item()
        j = torch.randint(0, w - tw + 1, size=(1,), generator=self.generator).item()
        return F.crop(img, i, j, th, tw)


    
class DeterministicRandomHorizontalFlip:
    def __init__(self, p, generator):
        self.p = p
        self.generator = generator

    def __call__(self, img):
        if torch.rand(1, generator=self.generator).item() < self.p:
            return F.hflip(img)
        return img

class DeterministicColorJitter:
    def __init__(self, brightness, contrast, saturation, hue, generator):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.generator = generator

    def __call__(self, img):
        fn_idx = torch.randperm(4, generator=self.generator)

        b = 1.0 + ((2 * torch.rand(1, generator=self.generator).item() - 1) * self.brightness) if self.brightness > 0 else 1.0
        c = 1.0 + ((2 * torch.rand(1, generator=self.generator).item() - 1) * self.contrast) if self.contrast > 0 else 1.0
        s = 1.0 + ((2 * torch.rand(1, generator=self.generator).item() - 1) * self.saturation) if self.saturation > 0 else 1.0
        h_shift = ((2 * torch.rand(1, generator=self.generator).item() - 1) * self.hue) if self.hue > 0 else 0.0

        for fn_id in fn_idx:
            fn = fn_id.item()
            if fn == 0 and self.brightness != 0:
                img = F.adjust_brightness(img, b)
            elif fn == 1 and self.contrast != 0:
                img = F.adjust_contrast(img, c)
            elif fn == 2 and self.saturation != 0:
                img = F.adjust_saturation(img, s)
            elif fn == 3 and self.hue != 0:
                img = F.adjust_hue(img, h_shift)

        return img


class Transformer():
    def __init__(self, seed, generator):
        self.seed = seed
        self.generator = generator
        self.for_dataset_type  = None
        self.__transforms_networks = {
            "IMAGENET": {"ResNet18": self.__get_transforms_IMAGENET_ResNet18},
            "MNIST": {"Net": self.__get_transforms_MNIST_Net, "Triplet_Net":self.__get_transforms_MNIST_Net},
        }

        self.__normalized = {
            "MNIST": {"mean": (0.1307, ), "std":(0.3081, )  },
            "IMAGENET": {"mean":(0.485, 0.456, 0.406), "std":(0.229, 0.224, 0.225) }
 
        }
    
    #---------------------------- FUNZIONI DETERMINISTICHE ----------------------------------------
    def deterministic_random_rotation(self,degrees,generator):
        return DeterministicRandomRotation(degrees,generator)

    
    def deterministic_random_horizontal_flip(self, p, generator):
        return DeterministicRandomHorizontalFlip(p, generator)

    def deterministic_color_jitter(self,  brightness, contrast, saturation, hue, generator):
        return DeterministicColorJitter(brightness, contrast, saturation, hue, generator)
    
    def deterministic_random_crop(self, size, padding, generator):
        return DeterministicRandomCrop(size, padding, generator)




    ##################################################################################
    def __get_transforms_IMAGENET_ResNet18(self, bool_all, bool_train):
        torch.manual_seed(42)
        transform = None
        if bool_all:
            transform = transforms.Compose([
              transforms.Resize((256,256)), 
              transforms.ToTensor() # 1) CONVERSIONE DA PIL a TENSOR
            ])          

        elif bool_train: # train
            transform = transforms.Compose([
            transforms.ToPILImage(), # 2) riconverto da Tensor a PIL  (serve per applicare le trasformazioni )
            transforms.RandomResizedCrop(224),  # Ritaglio casuale per aumentare la varietà e Ridimensiona il ritaglio alla dimensione 224x224.
            transforms.RandomHorizontalFlip(),  # Flip orizzontale casuale # trasformazioni di augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(), # 2) riconverto da PIL  Tensor (questo fa si che i valori siano riportati ell'intervallio [0-1])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        else:  # test & val
            transform = transforms.Compose([  
                transforms.ToPILImage(),
                transforms.CenterCrop(224),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
                

        return transform
    ##################################################################

    def __get_transforms_MNIST_Net(self, bool_all=None, bool_train=None):
        
        torch.manual_seed(42)
        transform = None
        if bool_all:
            transform = transforms.Compose([
              transforms.Grayscale(num_output_channels=1), # Converte forzamente in grayscale 
              transforms.Resize((28,28)), 
              transforms.ToTensor() # 1) CONVERSIONE DA PIL o numpy uint8 [h,w,c] a TENSOR -> [0,1] (1,28,28)
            ]) 
        elif bool_train:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))

            ])
        else:
            transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])
        return transform 

        ##################################################################



    
    

    
    def get_transforms(self, for_dataset, architecture, bool_all=None, bool_train=None):

        if self.__transforms_networks.get(f"{for_dataset}"):
            if self.__transforms_networks.get(f"{for_dataset}").get(f"{architecture}"):
                return self.__transforms_networks[f"{for_dataset}"][f"{architecture}"](bool_all,bool_train)
        else:

            print(f"Key '{for_dataset}.{architecture}' doesn't present in trasforms_dataset.", file=sys.stderr)
            raise KeyError(f"Key '{for_dataset}.{architecture}' doesn't present in trasforms_dataset.")
            

    def get_normalized(self, for_dataset):
        if self.__normalized.get(f"{for_dataset}"):
            mean = self.__normalized[f"{for_dataset}"]["mean"]
            std = self.__normalized[f"{for_dataset}"]["std"]
            return (mean, std)

        else:
            return None
        
    def set_for_dataset_type(self, for_dataset_type):
        self.for_dataset_type = for_dataset_type
    
