
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from abc import ABC, abstractmethod #Abstract Base Class
from colorama import Fore
import functools
class Architecture(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_criterion(self):
        pass

    def get_scheduler(self):
        pass

    def get_info_architecture(self):
        pass


class ResNet18_IMAGENET(nn.Module, Architecture):
    def __init__(self, num_classes, pretrained):
        super(ResNet18_IMAGENET, self).__init__()
        #--------------------
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # import ResNet18 pre-addestrata su ImageNet
        
        in_features = self.model.fc.in_features  # Get the number of inputs of the FC layer
        self.model.fc = nn.Linear(in_features, num_classes)  # # replace the FC layer
        
        
        self.criterion_name = None
        #-----------------------------------------------------------

        self.size_input_img = (3,224,224)

        self.lr = None # default
        self.weight_decay = 0 # default 
        self.momentum = 0

        self.batch_size = 0

        self.scheduler = None
        self.scheduler_name = None
        self.optimizer = None

        self.criterion = None
        


    def get_optimizer(self, step_size=30, gamma=0.1):

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        return self.optimizer


    def set_lr(self, lr):
        self.lr = lr

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_momentum(self, momentum):
        self.momentum = momentum

    def get_model(self):
        return self.model

    def get_criterion(self):
        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
            self.criterion_name = self.criterion.func.__name__  if isinstance(self.criterion, functools.partial) else self.criterion.__class__.__name__

        return self.criterion
    
    def set_criterion(self, loss_obj):
        self.criterion = loss_obj
        self.criterion_name = self.criterion.func.__name__  if isinstance(self.criterion, functools.partial) else self.criterion.__class__.__name__

    def get_criterion_name(self):
        return self.criterion_name

    def get_scheduler(self):
        return self.scheduler
    
    def  set_scheduler_name(self, name):
        self.scheduler_name = name
    
    def set_batch_size(self, bz):
        self.batch_size = bz

    def get_info_architecture(self):

        info_dict = { "optimizer": {"type": type(self.optimizer).__name__, "lr":self.lr, "momentum":self.momentum, "weight_decay": self.weight_decay},
                      "scheduler": {"type": type(self.scheduler).__name__ if self.scheduler is not None else  "false" },
                      "criterion": {"type": type(self.criterion).__name__},
                      "architecture_obj": self.__class__.__name__ 
                    }

        return info_dict

    def get_size_img_input(self): 
        return self.size_input_img



#####################################################################################################


class Net_MNIST(nn.Module, Architecture):

    
    def __init__(self, num_classes, pretrained):
        super(Net_MNIST, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)


        self.size_input_img = (1,28,28)

        self.lr = None # default
        self.weight_decay = 0 # default 
        self.momentum = 0

        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.criterion = None 
        self.criterion_name  = None


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.criterion_name == "nll_loss":
            x = F.log_softmax(x, dim=1)
        return x

        
    
    def set_lr(self, lr):
        self.lr = lr

    def get_optimizer(self, model):
        #self.model = self.get_model()
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum )
        return self.optimizer


    def get_criterion(self):
        if self.criterion is None:
            self.criterion =  nn.CrossEntropyLoss(reduction='mean')
            self.criterion_name = self.criterion.func.__name__  if isinstance(self.criterion, functools.partial) else self.criterion.__class__.__name__

        return self.criterion
    
    def set_criterion(self, loss_obj):
        self.criterion = loss_obj
        self.criterion_name = self.criterion.func.__name__  if isinstance(self.criterion, functools.partial) else self.criterion.__class__.__name__


    def get_criterion_name(self):
        return self.criterion_name
    
    def get_batch_size_origin(self):
        return 128

    def get_lr_origin(self):
        return 0.01

    def get_momentum_origin(self):
        return 0.5

    def get_info_architecture(self):

        info_dict = { "optimizer": {"type": type(self.optimizer).__name__, "lr":self.lr, "momentum":self.momentum, "weight_decay": self.weight_decay},
                      "scheduler": {"type": type(self.scheduler).__name__ if self.scheduler is not None else  "false" },
                      "criterion": {"type": self.criterion.func.__name__  if isinstance(self.criterion, functools.partial) else self.criterion.__class__.__name__ },
                      "architecture_obj": self.__class__.__name__ 
                    }

        return info_dict

    def get_size_img_input(self): 
        return self.size_input_img
    
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_momentum(self, momentum):
        self.momentum = momentum
    
    def get_scheduler(self):
        return self.scheduler
    
    def get_model(self):
        return self.model 
    
    def set_model(self, model):
        self.model = model
    
    def set_batch_size(self, bz):
        self.batch_size = bz




    def transfer_learning_feature_based(self,model,path_model_transfer,num_classes_tot ):
        print(f"transfer_learning_feature_based - Net MNIST - class num transfer {num_classes_tot}")
        model.load_state_dict(torch.load(path_model_transfer))
        #--------- Freeze all the weights of the feature extractor and replace the last layer
        for param in model.parameters():
            param.requires_grad = False 

        num_ftrs = model.fc2.in_features
        model.fc2 = nn.Linear(num_ftrs, num_classes_tot)




##################################################################################################################



class Manager_Networks():

    def __init__(self):

        self._collection_networks_pretrained = {
            "ResNet18": {"IMAGENET": self._get_ResNet18_IMAGENET}   
        }

        self._collection_networks_no_pretrained = {
            "Net": {"MNIST": self._get_Net_MNIST}
        }

        self._collection_architectures = {
            "ResNet18_IMAGENET": self._get_ResNet18_IMAGENET,
            "Net_MNIST": self._get_Net_MNIST
        }


        


    def _get_ResNet18_IMAGENET(self, num_classes, pretrained, dropout=None):
        return ResNet18_IMAGENET(num_classes,pretrained)
    
    def _get_Net_MNIST(self, num_classes, pretrained):
        return Net_MNIST(num_classes, pretrained)


    

    def get_network_pretrained(self, architecture, for_dataset, num_classes, dropout=None):
        print(f"Architecture {architecture}")
        if self._collection_networks_pretrained.get(f"{architecture}").get(f"{for_dataset}"):

            return self._collection_networks_pretrained[f"{architecture}"][(f"{for_dataset}")](num_classes, pretrained=True, dropout= dropout)
        else:
            raise KeyError(f"Key '{architecture}.{for_dataset}' doesn't present in collection_networks_pretrained.")
            print(f"Key '{architecture}.{for_dataset}' doesn't present in collection_networks_pretrained.", file=sys.stderr)
            return None
        
    def get_network_no_pretrained(self, architecture, for_dataset, num_classes, embedding_dim=None):
        if self._collection_networks_no_pretrained.get(f"{architecture}").get(f"{for_dataset}"):
            if embedding_dim is not None:
                return self._collection_networks_no_pretrained[f"{architecture}"][(f"{for_dataset}")](num_classes, embedding_dim, pretrained=False)
            else:
                return self._collection_networks_no_pretrained[f"{architecture}"][(f"{for_dataset}")](num_classes, pretrained=False)

        else:
            raise KeyError(f"Key '{architecture}.{for_dataset}' doesn't present in collection_networks_no_pretrained.")
            print(f"Key '{architecture}.{for_dataset}' doesn't present in collection_networks_no_pretrained.", file=sys.stderr)
            return None




    def get_architecture(self, name_architecture,num_classes, embedding_dim = None, pretrained=False):
        if self._collection_architectures.get(f"{name_architecture}"):
            if embedding_dim is not None:
                return self._collection_architectures[f"{name_architecture}"](num_classes, embedding_dim, pretrained)
            else:
                return self._collection_architectures[f"{name_architecture}"](num_classes, pretrained)

        else:
            raise KeyError(f"Key '{name_architecture}' doesn't present in collection_architectures")
            print(f"Key '{name_architecture}' doesn't present in collection_architecture", file=sys.stderr)
            return None


