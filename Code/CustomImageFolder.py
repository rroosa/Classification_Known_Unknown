import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import  collections 
from torch.utils.data import Subset 
from colorama import Fore
import sys
from collections import defaultdict
import random
import pandas as pd
from torchvision import datasets, transforms 
import torch

class CustomImageFolder(ImageFolder):
    def __init__(self, root, desiderated_classes, label_mapping = None, transform=None, target_transform=None, loader=default_loader,  transform_in_tensor=False):
        """
        Args:
            root: Path to the main dataset folder.  
            desired_classes: List of desired classes to include.  
            label_mapping: Map labels to new ones {label_old: label_new} # use only in cases of unknown classes  
            transform: Transformations to apply to the images.  
            target_transform: Transformations to apply to the labels (use only for dynamic transformations).  
            loader: Function to load images (default: PIL).
        """
        
        self.desiderated_classes = desiderated_classes
        self.class_name = []                                # List used to keep track of names (original class names)
        self.label_mapping = label_mapping 
        print("Desiderated classes",self.desiderated_classes)
        print(f"Custom Image Folder {transform}")

        if isinstance(transform, transforms.Compose):
            has_to_tensor = any(isinstance(t, transforms.ToTensor) for t in transform.transforms)
            if not has_to_tensor and (transform_in_tensor==True):
                #print("transform in tensor")
                transform = transforms.Compose( transform.transforms + [transforms.ToTensor()])
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader)
        self.idx_to_class = { idx: class_name for class_name ,idx in self.class_to_idx.items() }
        self.class_name = [ self.idx_to_class.get(label) for _, label  in self.samples ] # collected names
        self.targets = [ label for _, label in self.samples ] # raccolgo le etichette 

        ## apply the mapping, updating sample, class_to_idx, targets
        if self.label_mapping:
            self.class_to_idx = { class_name: self.label_mapping.get(idx, idx) for class_name ,idx in self.class_to_idx.items() }
            self.samples = [ (path, self.label_mapping.get(label, label)) for path, label in self.samples ]
            self.targets = [ s[1] for s in self.samples ] # here I do not need to apply the mapping because I am already acting on sample and the mapping has just been done


    def find_classes(self, directory): # redefine the ImageFolder function, called with super().__init__

        # Find only the classes specified in desired_classes to include for dataset formation
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name in self.desiderated_classes]  # List of desired folder names
        #classes.sort()  # Optionally modify here to permute the classes, i.e., instead of sorting, assign the desired_classes list directly
        if len(classes) == len(self.desiderated_classes) and set(classes).issubset(set(self.desiderated_classes)):
            classes = self.desiderated_classes              #It means that all the desired classes have been found; perform the assignment in a way that preserves the order of the class names.
        else:
            print(Fore.RED+f"Desiderated classes don't find in the folder {directory}"+ Fore.RESET)
            sys.exit()
                                                                                                
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes) }# creation of map {nome:id}
        
        
        return classes, self.class_to_idx

    def get_sample_info(self, index):
        path, label = self.samples[index]        # (PATH, LABEL (eventually edited))
        class_name  = self.class_name[index]     # (ORIGIN NAME) 
        #print(f" Targets: {self.targets[index]}, class_name: {class_name}, label: {label}, path_naem {path}")
        return path, label, class_name

    def get_list_targets(self):
        return self.targets 

    def counter_for_classes(self):
        class_counts = collections.Counter(self.targets)
        #print(class_counts)
        return class_counts
    

    

    




class CustomImageFolderSubset(Subset):

    def __init__(self, dataset, indices, transforms = None, label_mapping = None, extract_sample_for_class = None):
        #super().__init__(dataset, indices)
        self.dataset = dataset              #Initialize the reference to the source dataset.  
        self.label_mapping = label_mapping  # Use mapping to the subset (used only when the modification is to be applied exclusively to the train or test set)
        self.transforms = transforms
        if self.transforms is None:
            print("Transform non è presente!")
        else:
             print(f"Transform è presente! {transforms}")
        #print(f"CustomImageFolderSubset- transforms {transforms}")
        targets = [ self.dataset.targets[index_origin] for index_origin in indices ] # Labels associated with the samples in the subset
        if self.label_mapping is not None:
            self.targets = [ self.label_mapping.get(target, target) for target in targets]
        else:
            self.targets = targets
        print(f"CustomImageFolderSubset - Verifica lunghezza Indice:{len(indices)}- Target {len(self.targets)}")
       
        # if is specificated extract_sample_for_class, filtered the index
        if extract_sample_for_class is not None:
            print(f"Number of samples to extract {extract_sample_for_class}")
            class_to_indices = defaultdict(list)
            for idx, target in zip(indices, self.targets):
                class_to_indices[target].append(idx)
            
            #print(class_to_indices)

            selected_indices = []
            for target, list_idx in class_to_indices.items():
                available = len(list_idx)
                print(f"Classe [{target}]- num indici {available}")
                try:
                    if available < extract_sample_for_class:
                        raise ValueError(f"There are not enough samples for class {target}: available {available} < {extract_sample_for_class}")
                    selected_indices.extend(random.sample(list_idx, extract_sample_for_class))
                
                except ValueError as e:
                    print(f"Use all available ones {available}")
                    selected_indices.extend(random.sample(list_idx, available))




            indices = selected_indices  # aUpdate the indices of the subset
            print(f"num. tot of index {len(indices)}")

        super().__init__(dataset, indices)
        print(f"Total number of indices after super: {len(self.indices)} – target length not yet adjusted {len(self.targets)}")
        # Retrieve the final labels after filtering
        targets = [self.dataset.targets[i] for i in self.indices]
        if self.label_mapping is not None:
            self.targets = [self.label_mapping.get(t, t) for t in targets]
        else:
            self.targets = targets
        
        print(f"Total number of targets after adjustment - {len(self.targets)}")
        self.class_name = [ self.dataset.class_name[index_origin] for index_origin in self.indices]
        print(f"total number of class_names after adjustment - {len(self.class_name)}")

    def __getitem__(self, subset_index):
        
        image, label = self.dataset[self.indices[subset_index]] # call  function __getitem__ of CustomImageFolder
        
        if self.label_mapping is not None:
            label = self.label_mapping.get(label,label)
        

        if self.transforms is not None:
            
            image = self.transforms(image)



        return image, label
    
    def __len__(self):
        return len(self.targets)
    
    def get_sample_info(self, subset_index):
        original_index = self.indices[subset_index] 

        path, label, class_name =  self.dataset.get_sample_info(original_index)     # return path, label, class_name
        if self.label_mapping is not None:
            label = self.label_mapping.get(label, label) 
        return path, label, class_name

    def get_list_targets(self):
        return self.targets 
    
    def counter_for_classes(self):
        class_counts = collections.Counter(self.targets)
        #print(class_counts)
        return class_counts
    
    def compute_mean_dev(self):
        transform = transforms.Compose([transforms.ToTensor()])
        list_img = []
        for indice in self.indices:
            img, _ = self.dataset[indice]
            #img_tr = transform(img)
            list_img.append(img)

        all_imgs = torch.stack(list_img)
        mean = all_imgs.mean(dim=(0, 2, 3))  # (N, C, H, W)
        std  = all_imgs.std(dim=(0, 2, 3))

        print(f"Mean {mean}")
        print(f"Dev std {std}")


    

    def get_dict_path_img(self, key):
        self.list_path_img = []
        for subset_index  in self.indices:
            
            path, label, class_name =  self.dataset.get_sample_info(subset_index)
            ##############
            dict_path = {"dataset": key, "filename_img": path, "label":label, "class_name": class_name}
            self.list_path_img.append(dict_path) 
            #############
        
        df = pd.DataFrame(self.list_path_img)
        return df

    


