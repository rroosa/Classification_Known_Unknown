import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import Subset 
from PIL import Image
from collections import defaultdict
import numpy as np
import random
import collections
from dataset_collection import *
from colorama import Fore
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CustomDatasetKnown(Dataset):
                                                            # modificare facendo in modo che passo transform_all e transform_train o transform test
    def __init__(self, root, classes_desiderate,num_for_class = None, replace_label=None, bool_train=True, transform=None, save_dir="./filtered_mnist", type_dataset = "MNIST"):
        self.name_dataset = type_dataset
        self.dataset = collection_datasets[type_dataset](root, bool_train, transform)
        #self.dataset = datasets.MNIST(root=root, train=train, download=True)

        #loader = DataLoader(self.dataset, batch_size=500, shuffle=False, num_workers=2)

        self.transform = transform
        self.save_dir = save_dir  # Directory per salvare le immagini
        self.replace_label = replace_label  
        self.min_samples = None
        self.desiderated_classes = classes_desiderate
        print(f"CustomDatasetKnown - {self.replace_label} ")
        
        # Creare la cartella se non esiste
        os.makedirs(self.save_dir, exist_ok=True)
        if bool_train == True:
            key ="train_valid"
            # salva i percorsi delle immagini di train e validation in un file csv ( class - path )
            #file_csv_of_path = f"{type_dataset}_path_img_train_valid.csv"
        else:
            key = "test"
            #file_csv_of_path = f"{type_dataset}_path_img_test.csv"

        

        self.image_paths = []  # Per salvare i percorsi reali
        self.targets  = []     # Per tenere traccia delle etichette date da 0 .... 
        self.class_name = [] # Per tenere traccia dei nomi delle classi
        class_indices = defaultdict(list)

        self.class_to_idx = {class_name:idx for idx, class_name in enumerate(self.desiderated_classes) } # utilizzato per ripiazzare i nomi delle cartelle con targert a partire da 0 ...
        self.idx_to_class = { idx: class_name for class_name ,idx in self.class_to_idx.items() }         # utilizzato per tenere traccia dei nomi delle cartelle, una volta che si è assegnato il target
        print(self.class_to_idx)
        print(self.idx_to_class)
        class_to_idx_origin = self.dataset.class_to_idx    
        print(Fore.GREEN+f"{self.dataset.class_to_idx }"+Fore.RESET) #{'0 - zero': 0, '1 - one': 1, '2 - two': 2, '3 - three': 3, '4 - four': 4, '5 - five': 5, '6 - six': 6, '7 - seven': 7, '8 - eight': 8, '9 - nine': 9}
        idx_to_class_origin = { idx: class_n for class_n, idx in class_to_idx_origin.items()}
        #nomi_cartelle_classi = list(self.dataset.class_to_idx.keys())
        for cartella in classes_desiderate:
            path_cartella = os.path.join(self.save_dir, cartella)
            print(path_cartella)
            os.makedirs(path_cartella, exist_ok=True)
        # Filtrare e salvare solo le classi 2,4,6,8
        for i in range(len(self.dataset)):
            img = self.dataset.data[i]
            if isinstance(self.dataset.targets, torch.Tensor ):
                target = self.dataset.targets[i].item()
            else:
                target = self.dataset.targets[i]
            #print(f"Traget {type(target)} {target}")
            name_class = idx_to_class_origin.get(target)  # vale solo per MNIST
            #print(f"Name classes {name_class}")
            if name_class in self.desiderated_classes: # [2,4,8,6]
                # Salvare l'immagine in PNG
                file_name = f"image_{i}_target_[{name_class}].png" # label è il nome dela cartella
                file_path = os.path.join(self.save_dir, name_class)
                file_path = os.path.join(file_path, file_name)
                #print(f"salve {file_name}")


                #print(Fore.RED+f"Dove salvare le immagini Del subset {file_path}"+Fore.RESET)
                # Convertire l'immagine in PIL e salvarla
                img = img if isinstance(img, np.ndarray) else img.numpy()
                #print(f"Salva immagini in percorso {file_path}")
                #print(f"Tipo {type(img)}")
                #print(f"Dimensioni {img.shape}") # 28 * 28

                img_pil = Image.fromarray(img)
                img_pil.save(file_path)

                # Memorizzare il percorso, il target e il nome della cartella
                self.image_paths.append(file_path)
                self.targets.append(self.class_to_idx.get(name_class))
                self.class_name.append(name_class)
                class_indices[name_class].append(len(self.image_paths) - 1)
        
        #print(f"{self.targets}\n {self.class_name}")
        # **Bilanciare le classi**
        if num_for_class is None:
            self.min_samples = min(len(class_indices[c]) for c in self.desiderated_classes)
            print(f"Numero minimo di campioni{self.min_samples}")

        balanced_indices = []
        for c in self.desiderated_classes:
            if num_for_class is not None:
                self.min_samples = min(num_for_class, len(class_indices[c]))
                print(f"Numero minimo di campioni{self.min_samples}")
            #print(f"Entra {c}")
            balanced_indices.extend(np.random.choice(class_indices[c], self.min_samples, replace=False).tolist())

        # aggiorno i percorsi selezionati, i target e i nomi
        self.image_paths = [self.image_paths[i] for i in balanced_indices]
        self.targets = [self.targets[i] for i in balanced_indices]
        self.class_name = [ self.class_name[i] for i in balanced_indices]
        # aggiono il dataset , aggiorno gli indici che adesso partiranno da 0
        #self.dataset = self.dataset[balanced_indices]
        self.indices = [i for i in range(len(self.class_name))] 
        print(f"TOT {len(self.targets)} ==  {len( self.indices)} == {len(self.class_name)} == {len(self.image_paths)} == {len(balanced_indices)} ")
        
        if replace_label is not None:
            self.targets = [self.replace_label.get(self.targets[i]) for i in self.indices]
            self.class_to_idx = {class_name:self.replace_label.get(idx) for idx, class_name in enumerate(self.desiderated_classes) } # utilizzato per ripiazzare i nomi delle cartelle con targert a partire da 0 ...
            self.idx_to_class = { idx: class_name for class_name ,idx in self.class_to_idx.items() }  
            print(Fore.MAGENTA+f"Dopo il labelling {self.class_to_idx} ----- {self.idx_to_class}")
            


        
        print(f"TOT {len(self.targets)} ==  {len( self.indices)} == {len(self.class_name)} == {len(self.image_paths)} ")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Caricare l'immagine dal file
        img = Image.open(self.image_paths[idx])
        real_label = self.targets[idx] # l'etichetta parte già da 0,1,2,3 o in caso di unknown solo 4

        # Applicare le trasformazioni se definite
        if self.transform:
            print("Custom dataset known" )
            img = self.transform(img)

        return img, real_label
       

    def get_item_path(self, idx):
        path = self.image_paths[idx] 
        real_label =  self.targets[idx]
        class_name =  self.class_name[idx] 

        return path, real_label, class_name
    
    def get_sample_info(self, idx):
        path = self.image_paths[idx] 
        real_label =  self.targets[idx]
        class_name =  self.class_name[idx] 

        return path, real_label, class_name

    def get_num_sample_for_classes(self):
        return self.min_samples

    def counter_for_classes(self):
        class_counts = collections.Counter(self.targets)
        #print(class_counts)
        return class_counts

    def get_list_targets(self):

        return self.targets



class CustomDatasetUnknown(Dataset):
        
    def __init__(self, root, classes_desiderate,num_for_class = None, replace_label=None, bool_train=True, transform=None, save_dir="./filtered_mnist", type_dataset = "MNIST"):
        print(f"CustomDatasetUnknown ")
        self.name_dataset = type_dataset
        self.dataset = collection_datasets[type_dataset](root, bool_train, transform)
        
        self.transform = transform
        self.save_dir = save_dir  # Directory per salvare le immagini
        self.replace_label = replace_label  
        self.min_samples = None
        self.desiderated_classes = classes_desiderate
        print(f"Replace label {self.replace_label}- transform {transform}")
        
        # Creare la cartella se non esiste
        os.makedirs(self.save_dir, exist_ok=True)
        if bool_train == True:
            key ="train_valid"
            # salva i percorsi delle immagini di train e validation in un file csv ( class - path )
            #file_csv_of_path = f"{type_dataset}_path_img_train_valid.csv"
        else:
            key = "test"
            #file_csv_of_path = f"{type_dataset}_path_img_test.csv"

        

        self.image_paths = []  # Per salvare i percorsi reali
        self.targets  = []     # Per tenere traccia delle etichette date da 0 .... 
        self.class_name = [] # Per tenere traccia dei nomi delle classi
        class_indices = defaultdict(list)

        self.class_to_idx = {class_name:idx for idx, class_name in enumerate(self.desiderated_classes) } # utilizzato per ripiazzare i nomi delle cartelle con targert a partire da 0 ...
        self.idx_to_class = { idx: class_name for class_name ,idx in self.class_to_idx.items() }         # utilizzato per tenere traccia dei nomi delle cartelle, una volta che si è assegnato il target
        print(self.class_to_idx)
        
        #print(self.idx_to_class)
        if hasattr(self.dataset, 'class_to_idx'):
            class_to_idx_origin = self.dataset.class_to_idx    
            print(Fore.GREEN+f"{self.dataset.class_to_idx }"+Fore.RESET) #{'0 - zero': 0, '1 - one': 1, '2 - two': 2, '3 - three': 3, '4 - four': 4, '5 - five': 5, '6 - six': 6, '7 - seven': 7, '8 - eight': 8, '9 - nine': 9}
            idx_to_class_origin = { idx: class_n for class_n, idx in class_to_idx_origin.items()}
            print(f"Mappa class to idx origin {class_to_idx_origin}")
        else:
            
            class_to_idx_origin = {str(i): i for i in range(10)}
            idx_to_class_origin = { idx: class_n for class_n, idx in class_to_idx_origin.items()}

        if hasattr(self.dataset, 'labels'):
            self.dataset.targets = self.dataset.labels


        
        #nomi_cartelle_classi = list(self.dataset.class_to_idx.keys())
        for cartella in classes_desiderate:
            path_cartella = os.path.join(self.save_dir, cartella)
            #print(path_cartella)
            os.makedirs(path_cartella, exist_ok=True)
        # Filtrare e salvare solo le classi 2,4,6,8
        for i in range(len(self.dataset)):
            img = self.dataset.data[i]
            if isinstance(self.dataset.targets, torch.Tensor ):
                target = self.dataset.targets[i].item()
            else:
                target = self.dataset.targets[i]
            #print(f"Traget {type(target)} {target}")
            name_class = idx_to_class_origin.get(target)  # vale solo per MNIST
            #print(f"Name classes {name_class}")
            if name_class in self.desiderated_classes: # [2,4,8,6]
                # Salvare l'immagine in PNG
                file_name = f"image_{i}_target_[{name_class}].png" # label è il nome dela cartella
                file_path = os.path.join(self.save_dir, name_class)
                file_path = os.path.join(file_path, file_name)
                #print(f"salve {file_path}")

                #print(f"Tipo {type(img)}")
                #print(f"Dimensioni {img.shape}") # 28 * 28
                #print(Fore.RED+f"Dove salvare le immagini Del subset {file_path}"+Fore.RESET)
                if isinstance(img, np.ndarray) and (img.shape[0] == 3 or img.shape[0] == 1): # necessario per SVNH
                    # effettua la trasposta (3,32,32) --> (32,32,3)
                    img = np.transpose(img, (1, 2, 0))
                    # Convertila in uint8 se necessario
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)

                # Convertire l'immagine in PIL e salvarla
                img = img if isinstance(img, np.ndarray) else img.numpy()
                #print(f"Salva immagini in percorso {file_path}")
                #print(f"Tipo {type(img)}")
                #print(f"Dimensioni {img.shape}") # 28 * 28

                img_pil = Image.fromarray(img)
                img_pil.save(file_path)

                # Memorizzare il percorso, il target e il nome della cartella
                self.image_paths.append(file_path)
                self.targets.append(self.class_to_idx.get(name_class))
                self.class_name.append(name_class)
                class_indices[name_class].append(len(self.image_paths) - 1)
        
        #print(f"{self.targets}\n {self.class_name}")
        # **Bilanciare le classi**
        if num_for_class is None:
            self.min_samples = min(len(class_indices[c]) for c in self.desiderated_classes)
            #print(f"Numero minimo di campioni{self.min_samples}")

        balanced_indices = []
        for c in self.desiderated_classes:
            if num_for_class is not None:
                self.min_samples = min(num_for_class, len(class_indices[c]))
                print(f"Numero minimo di campioni{self.min_samples}")
            #print(f"Entra {c}")
            balanced_indices.extend(np.random.choice(class_indices[c], self.min_samples, replace=False).tolist())

        # aggiorno i percorsi selezionati, i target e i nomi
        self.image_paths = [self.image_paths[i] for i in balanced_indices]
        self.targets = [self.targets[i] for i in balanced_indices]
        self.targets_balanced = self.targets
        self.class_name = [ self.class_name[i] for i in balanced_indices]
        class_counts = collections.Counter(self.targets_balanced)
        #print(Fore.RED+f"! CONTEGGIO PER CLASSE targets_balanced {class_counts}"+Fore.RESET)
        # aggiono il dataset , aggiorno gli indici che adesso partiranno da 0
        #self.dataset = self.dataset[balanced_indices]
        self.indices = [i for i in range(len(self.class_name))] 
        print(f"TOT {len(self.targets)} ==  {len( self.indices)} == {len(self.class_name)} == {len(self.image_paths)} == {len(balanced_indices)} ")
        
        if replace_label is not None:
            self.targets = [self.replace_label.get(self.targets[i]) for i in self.indices]
        
        class_counts = collections.Counter(self.targets)
        #print(Fore.RED+f"! CONTEGGIO PER CLASSE targets {class_counts}"+Fore.RESET)
        print(f"TOT {len(self.targets)} ==  {len( self.indices)} == {len(self.class_name)} == {len(self.image_paths)} ")
        class_counts = collections.Counter(self.targets_balanced)
        #print(Fore.RED+f"! CONTEGGIO PER CLASSE targets_balanced FOPP {class_counts}"+Fore.RESET)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Caricare l'immagine dal file
        img = Image.open(self.image_paths[idx])
        real_label = self.targets[idx] # l'etichetta parte già da 0,1,2,3 o in caso di unknown solo 4

        # Applicare le trasformazioni se definite
        if self.transform:
            img = self.transform(img)

        return img, real_label
       

    def get_item_path(self, idx):
        path = self.image_paths[idx] 
        real_label =  self.targets[idx]
        class_name =  self.class_name[idx] 

        return path, real_label, class_name
    
    def get_sample_info(self, idx):
        path = self.image_paths[idx] 
        real_label =  self.targets[idx]
        class_name =  self.class_name[idx] 

        return path, real_label, class_name

    def get_num_sample_for_classes(self):
        return self.min_samples

    def counter_for_classes(self):
        class_counts = collections.Counter(self.targets_balanced)
        #print(Fore.RED+f"! CONTEGGIO PER CLASSE {class_counts}"+Fore.RESET)
        return class_counts

    def get_list_targets(self):

        return self.targets



class CustomSubset(Dataset):
    
    def __init__(self, dataset, indices, transforms = None, label_mapping = None):
        
        self.dataset = dataset              #Initialize the reference to the source dataset
        self.label_mapping = label_mapping  #Use of mapping to the subset (used only when you want to apply the change only to the train or test set)
        self.transforms = transforms
        self.indices = indices
        targets = [ self.dataset.targets[index_origin] for index_origin in self.indices ] #Labels associated with the samples of the subset
        if self.label_mapping is not None:
            self.targets = [ self.label_mapping.get(target, target) for target in targets]
        else:
            self.targets = targets

        self.class_name = [ self.dataset.class_name[index_origin] for index_origin in self.indices]

        self.list_path_img = []

        

    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, subset_index):
        
        image, label = self.dataset[self.indices[subset_index]] # call function __getitem__ 
                
        if self.label_mapping is not None:
            label = self.label_mapping.get(label,label)

        if self.transforms is not None:
            image = self.transforms(image)

        
        return image, label
    

   

    
    def get_sample_info(self, subset_index):
        original_index = self.indices[subset_index] 

        path, label, class_name =  self.dataset.get_sample_info(original_index)     # returns path, label, class_name
        if self.label_mapping is not None:
            label = self.label_mapping.get(label, label) 
        return path, label, class_name

    def get_list_targets(self):
        return self.targets 
    
    def counter_for_classes(self):
        
        class_counts = collections.Counter(self.targets)
        #print(class_counts)
        return class_counts

    def get_dict_path_img(self, key):
        self.list_path_img = []
        for subset_index  in range(len(self.indices)):
            original_index = self.indices[subset_index] 
            path, label, class_name =  self.dataset.get_sample_info(original_index)
            ##############
            dict_path = {"dataset": key, "filename_img": path, "label":label, "class_name": class_name}
            self.list_path_img.append(dict_path) 
            #############
        
        df = pd.DataFrame(self.list_path_img)
        return df



    
