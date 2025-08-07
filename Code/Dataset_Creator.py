import numpy as np 
import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
import sys
import collections
from sklearn.model_selection import train_test_split
import random
#from Transformer import TransformedSubset, TransformedSubsetExtract
from dataset_collection import *
from colorama import Fore
from collections import Counter
import os
from torchvision.datasets import ImageFolder 
import pandas as pd
from CustomDataset import CustomDatasetKnown, CustomDatasetUnknown , CustomSubset                        # per dataset proveniete dalla libreria datasets
from CustomImageFolder import CustomImageFolder, CustomImageFolderSubset     # per dataset creati da ImageFolder
from ShuffledConcatDataset import ShuffledConcatDataset


class Dataset_Creator():
    def __init__(self, seed, generator):
        
        self.seed = seed
        self.generator = generator
        ########----------------------------------------####################
        ########               KNOWN                    ####################
        ########----------------------------------------####################

        self.dataset_all_known = None
        self.targets_all_known = None #list of target obtained after the balanced classes
        
        self.train_Y_known = None
        self.train_known = None # composed by  images and labels
        self.train_known_loader = None

        self.test_Y_known = None
        self.test_known = None #  composed by  images and labels
        self.test_known_loader = None



        self.validation_Y_known = None
        self.validation_known = None # composed by  images and labels
        self.validation_known_loader = None

        self.overall_Y_known = None 
        self.train_validation_Y_known = None

        self.known_assign_place = None

        #-------- RECIPROCAL ---------------
        self.train_reciprocal = None 
        self.validation_reciprocal = None 
        self.test_reciprocal = None  
        self.train_Y_reciprocal = None 
        self.validation_Y_reciprocal = None 
        self.test_Y_reciprocal = None 
        self.targets_all_reciprocal = None

        self.reciprocal_assign_place = None

        #-------- KNOWN_RECIPROCAL -----------
        self.train_known_reciprocal = None 
        self.validation_known_reciprocal = None  
        self.test_known_reciprocal = None 

        self.train_Y_known_reciprocal = None
        self.validation_Y_known_reciprocal = None 
        self.test_Y_known_reciprocal = None
        self.targets_all_known_reciprocal = None 

        self.train_known_reciprocal_loader = None 
        self.validation_known_reciprocal_loader = None 
        self.test_known_reciprocal_loader = None

        

        self.overall_Y_reciprocal = None 
        self.train_validation_Y_reciprocal = None

        #--------------UNKNOWN --------------------------
        self.test_Y_unknown = None
        self.test_unknown = None # composed by  images and labels

        self.test_known_reciprocal_unknown_loader = None


        self.path_folder_photos_known = None            # path of folder for phosos of known classes
        self.path_folder_pattern = None
        self.path_folder_photos_unknown = None 
        

        self.known_class_name = []                       # list of desiderated known classes
        self.reciprocal_class_name = []
        self.unknown_class_name = []
        

        self.known_map_name_to_label = None
        self.known_map_label_to_name = None 

        self.reciprocal_map_name_to_label = None 
        self.reciprocal_map_label_to_name = None

        self.known_reciprocal_map_label_to_name = None 
        self.known_reciprocal_map_name_to_label = None

        self.unknown_map_name_to_label = None

        self.test_Y_known_reciprocal_unknow = None
        self.test_known_reciprocal_unknown = None
        self.known_reciprocal_unknown_map_name_to_label = None



    
    def setFolder_Photos_Known(self, path):
        self.path_folder_photos_known = path
    
    def setFolder_Pattern(self, path):
        self.path_folder_pattern = path 
    
    def setFolder_Photos_Unknown(self, path):
        self.path_folder_photos_unknown = path
    
    def set_known_assign_place(self,known_assign_place):
        self.known_assign_place = known_assign_place 
    
    def set_reciprocal_assign_place(self, reciprocal_assign_place):
        self.reciprocal_assign_place = reciprocal_assign_place

    
    def add_known_class_name(self, class_name):
        self.known_class_name.append(class_name)
    

    def add_reciprocal_class_name(self, pattern_name ):
        self.reciprocal_class_name.append(pattern_name)
    
    def add_unknown_class_name(self, class_name):
        self.unknown_class_name.append(class_name)

    def get_known_map_name_to_label(self):
        return self.known_map_name_to_label
    
    def get_known_map_label_to_name(self):
        return self.known_map_label_to_name 
    
    def get_reciprocal_map_name_to_label(self):
        return self.reciprocal_map_name_to_label
    
    def get_reciprocal_map_label_to_name(self):
        return self.reciprocal_map_label_to_name
    
    def get_known_reciprocal_map_name_to_label(self):
        return self.known_reciprocal_map_name_to_label
    
    def get_known_reciprocal_map_label_to_name(self):
        return self.known_reciprocal_map_label_to_name
    
    def get_unknown_map_name_to_label(self):
        return self.unknown_map_name_to_label
    
    def get_known_reciprocal_unknown_map_name_to_label(self):
        return self.known_reciprocal_unknown_map_name_to_label
    

    def get_len(self, type_dataset):
        if type_dataset == 'train_known':
            return len(self.train_Y_known)

        elif type_dataset == 'validation_known':
            return len(self.validation_Y_known)

        elif type_dataset == 'test_known':
            return len(self.test_Y_known)

        elif type_dataset == 'test_unknown':
            return len(self.test_Y_unknown)

    
    def count_sample_in_classes(self, type_dataset=None):
        if type_dataset == "known_overall":
           
            #class_counts = collections.Counter(self.targets_all_known)
            class_counts = collections.Counter(self.overall_Y_known)
        
        elif type_dataset == "known_train_validation":
            class_counts = collections.Counter(self.train_validation_Y_known)

        elif type_dataset == "known_train":
            class_counts = self.train_known.counter_for_classes()
           
        elif type_dataset == "known_test":
            class_counts = self.test_known.counter_for_classes()
        
        elif type_dataset == "known_validation":
            class_counts = self.validation_known.counter_for_classes()

        elif type_dataset == "reciprocal_train_validation":
            class_counts = collections.Counter(self.targets_all_reciprocal) 
            
        elif type_dataset == "reciprocal_overall":
            class_counts = collections.Counter(self.targets_all_reciprocal)
            
        elif type_dataset == "reciprocal_train":
            class_counts = self.train_reciprocal.counter_for_classes()

        elif type_dataset == "reciprocal_validation":
            class_counts = self.validation_reciprocal.counter_for_classes()
           
        elif type_dataset == "reciprocal_test":
            class_counts = self.test_reciprocal.counter_for_classes()
        
        elif type_dataset == "unknown_test":
            class_counts = self.test_unknown.counter_for_classes()
        return class_counts

    

    def get_sample_info(self, type_dataset, idx):
        try:
            #------------------- KNOWN ---------------------------------------------
            if type_dataset == 'train_known':
                path, real_label, class_name = self.train_known.get_sample_info(idx)
            elif type_dataset == 'validation_known':
                path, real_label, class_name = self.validation_known.get_sample_info(idx)
                
            elif type_dataset == 'test_known':
                path, real_label, class_name = self.test_known.get_sample_info(idx)
            #------------------ RECIPROCAL ----------------------------------------
            elif type_dataset == 'train_reciprocal':
                path, real_label, class_name = self.train_reciprocal.get_sample_info(idx)
            elif type_dataset == 'validation_reciprocal':
                path, real_label, class_name = self.validation_reciprocal.get_sample_info(idx)
                
            elif type_dataset == 'test_reciprocal':
                path, real_label, class_name = self.test_reciprocal.get_sample_info(idx)
            
            #------------------ KNWON & RECIPROCAL ----------------------------------------
            elif type_dataset == 'train_known_reciprocal':
                path, real_label, class_name = self.train_known_reciprocal.get_sample_info(idx)
            elif type_dataset == 'validation_known_reciprocal':
                path, real_label, class_name = self.validation_known_reciprocal.get_sample_info(idx)
                
            elif type_dataset == 'test_known_reciprocal':
                path, real_label, class_name = self.test_known_reciprocal.get_sample_info(idx)
            
            #------------------ UKNOWN -----------------------------------------------------------
            elif type_dataset == 'test_unknown':
                path, real_label, class_name = self.test_unknown.get_sample_info(idx)
            
            #----------------- KNOWN & RECIPROCAL & UNKNOWN--------------------------------------
            elif type_dataset == 'test_known_reciprocal_unknown':
                path, real_label, class_name = self.test_known_reciprocal_unknown.get_sample_info(idx)
            
            return path, real_label, class_name

        except Exception as e:
            print(Fore.YELLOW+f"Error in get_sample_info {e}"+ Fore.RESET)
            return None, None, None
    
    def get_dict_path_img(self, type_dataset):
        if type_dataset == "train_known":
            df = self.train_known.get_dict_path_img("train")
        elif type_dataset == "validation_known":
            df = self.validation_known.get_dict_path_img("validation")
        elif  type_dataset == "test_known":
             df = self.test_known.get_dict_path_img("test")
        
        return df
    
    def get_concat_path(self, type_dataset, idx):

        if type_dataset == "train_known_reciprocal":

            """Find the correct path of image belongs to train_known_reciprocal ConcatDataset"""
            for d in self.train_known_reciprocal.datasets:  # Scorriamo i subset
                if idx < len(d):  # Se l'indice è all'interno di questo subset
                    path, real_label , name = d.get_sample_info(idx)
                    return path, real_label, name  # Restituisce il percorso originale
                idx -= len(d)  # Sottraiamo la lunghezza del subset corrente
            raise IndexError("Index out of dataset length")
        
        elif type_dataset == "validation_known_reciprocal":

            """Find the correct path of image belongs to validation_known_reciprocal ConcatDataset"""
            for d in self.validation_known_reciprocal.datasets:  # scroll the subsets
                if idx < len(d):  # if index is in this subset
                    path, real_label , name = d.get_sample_info(idx)
                    return path, real_label, name  # return the origin path
                idx -= len(d)  # subtract the length of the current subset
            raise IndexError("Index out of dataset length")
        
        elif type_dataset == "test_known_reciprocal":

            """Find the correct path of image belongs to test_known_reciprocal ConcatDataset"""
            for d in self.test_known_reciprocal.datasets:  # scroll the subsets
                if idx < len(d):  # if index is in this subset
                    path, real_label , name = d.get_sample_info(idx)
                    return path, real_label, name  #  return the origin path
                idx -= len(d)  # subtract the length of the current subset
            raise IndexError("Indice fuori dal range del dataset")

        elif type_dataset == "open_set_testing":

            """Find the correct path of image belongs to ConcatDataset"""
            for d in self.test_known_unknown.datasets:  # scroll the subsets
                if idx < len(d):  # if index is in this subset
                    path, real_label , name = d.get_sample_info(idx)
                    return path, real_label, name  #  return the origin path
                idx -= len(d)  # subtract the length of the current subset
            raise IndexError("Indice fuori dal range del dataset")
            
    def seed_worker_data_loader(self,worker_id):
        
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def create_DataLoader(self, type_dataset, batch_size ):


        if type_dataset == 'train_known':
            print("DataLoader-train-known")
            
            self.train_known_loader =  DataLoader(self.train_known, batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=self.generator, worker_init_fn=self.seed_worker_data_loader )
        
        elif type_dataset == 'validation_known':
            print("DataLoader-validation-known")
            self.validation_known_loader = DataLoader(self.validation_known, batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True,generator=self.generator , worker_init_fn=self.seed_worker_data_loader  )

        elif type_dataset == 'test_known':
            print("DataLoader-test-known")
            self.test_known_loader =  DataLoader(self.test_known, batch_size = batch_size, shuffle=False, num_workers=0)
        
        elif type_dataset == 'train_known_reciprocal':
            self.generator = torch.Generator()
            self.generator.manual_seed(42)
            print("DataLoader-train-known-reciprocal")
            self.train_known_reciprocal_loader =  DataLoader(self.train_known_reciprocal, batch_size = batch_size, shuffle = False, num_workers=0, pin_memory=True, generator=self.generator, worker_init_fn=self.seed_worker_data_loader )
        
        elif type_dataset == 'train_reciprocal':
            self.generator = torch.Generator()
            self.generator.manual_seed(42)
            print("DataLoader-train-reciprocal")
            self.train_reciprocal_loader =  DataLoader(self.train_reciprocal, batch_size = batch_size, shuffle = False, num_workers=0, pin_memory=True, generator=self.generator, worker_init_fn=self.seed_worker_data_loader )
        

        elif type_dataset == 'validation_known_reciprocal':
            self.generator = torch.Generator()
            self.generator.manual_seed(42)
            print("DataLoader-validation-known-reciprocal")
            self.validation_known_reciprocal_loader =  DataLoader(self.validation_known_reciprocal, batch_size = batch_size, shuffle = False, num_workers=0, pin_memory=True, generator=self.generator, worker_init_fn=self.seed_worker_data_loader )
        
        elif type_dataset == 'test_known_reciprocal':
            self.generator = torch.Generator()
            self.generator.manual_seed(42)
            print("DataLoader-test-known-reciprocal")
            self.test_known_reciprocal_loader =  DataLoader(self.test_known_reciprocal, batch_size = batch_size, shuffle = False, num_workers=0, pin_memory=True, generator=self.generator, worker_init_fn=self.seed_worker_data_loader)

        elif type_dataset == 'test_known_reciprocal_unknown':
            self.generator = torch.Generator()
            self.generator.manual_seed(42)
            print("DataLoader-test-known-reciprocal-unknown")
            self.test_known_reciprocal_unknown_loader =  DataLoader(self.test_known_reciprocal_unknown, batch_size = batch_size, shuffle = False, num_workers=0, pin_memory=True, generator=self.generator, worker_init_fn=self.seed_worker_data_loader)



    def get_datasetLoader(self, type_dataset):
        if type_dataset == 'train_known':
            print("get train load")
            return self.train_known_loader

        elif type_dataset == 'test_known':
            print("get test load")
            return self.test_known_loader

        elif type_dataset == 'validation_known':
            print("get valid load")
            return self.validation_known_loader
        
        elif type_dataset == 'train_known_reciprocal':
            print("get train k-r load")
            return self.train_known_reciprocal_loader

        elif type_dataset == 'test_known_reciprocal':
            print("get test k-r load")
            return self.test_known_reciprocal_loader

        elif type_dataset == 'validation_known_reciprocal':
            print("get valid k-r load")
            return self.validation_known_reciprocal_loader
        
        elif type_dataset == "test_known_reciprocal_unknown":
            print("get  loader open set testing")
            return self.test_known_reciprocal_unknown_loader

        elif type_dataset == 'train_reciprocal':
            print("get train k-r load")
            return self.train_reciprocal_loader
        


    def stratified_split_DatasetImageFolder(self, dataset, type_task, transforms_train, transforms_test,  validation_size, test_size, bool_balance = True, num_el_for_class = None, extract_sample_for_class_test= None):
        print(f"[X] stratified_split_DatasetImageFolder -  num_el_for_class {num_el_for_class}")
    
        # Perform a stratified split on a dataset.
        # dataset -> istance of ImageFolder or its subclasses
        targets = np.array(dataset.targets)
        indices = np.array([ i for i in range(len(targets)) ]) # Array of indices from 0 to length - 1


        # Check the number of elements per class
        class_counts = collections.Counter(targets.tolist())
        print(Fore.MAGENTA+f"Num sample for classes {class_counts}"+Fore.RESET)
        min_num_samples = min(class_counts.values()) 
        print(Fore.MAGENTA+f"Min num for class {min_num_samples}"+Fore.RESET)


        # For each class, collect the indices.
        class_counts = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in class_counts:
                    class_counts[label] = []
            class_counts[label].append(idx)
        

        if num_el_for_class is not None and isinstance(num_el_for_class, int) and bool_balance: # desired number of elements per class
            print(Fore.RED+f"Minimum expected number per class {num_el_for_class}")
            if num_el_for_class < min_num_samples: # If it is less than the actual value, take the actual value.
                print(Fore.RED+f"Minimum expected number per class {num_el_for_class} < effective  {min_num_samples}")
                min_num_samples = num_el_for_class 

      
        if bool_balance == True: 
            balanced_indices = []
            print(Fore.RED+f"Bilanciamento e selezione casuale")
            # Randomly select the same number of images for each class.
            for indices in class_counts.values():
                balanced_indices.extend(random.sample(indices, min_num_samples ))
            # add index
            indices = np.array(balanced_indices)
            # aggiornare i target 
            print(Fore.MAGENTA+f"Num tot indice {len(indices)}"+Fore.RESET)
            targets = targets[indices] # update all dataset targets to consider
            print(Fore.MAGENTA+f"Num tot TARGETS {len(targets)}"+Fore.RESET)
        
        elif bool_balance == False and num_el_for_class is not None and isinstance(num_el_for_class, dict):

            
            if len(class_counts) == 1 or len(class_counts) == 2: # altrimenti se il numero dei reciproci è solo 1 (reciprocal_all) prendi la media di campioni di known
                unbalanced_indices = [] 
                print(Fore.BLUE+f"CASO 2: #classi reciproci = 1 "+Fore.RESET)
                print(Fore.RED+f"Selezione del numero di pattern = al numero medio delle classi note"+Fore.RESET)
                print(f"Dizionario dei conteggi delle classi note {num_el_for_class} ")
                print(f"Collezione indici per classe pattern {class_counts.keys()} ")

                for label, array_indices in class_counts.items():
                    num_samples = int(sum(num_el_for_class.values())/len(num_el_for_class))
                    extracts_samples = num_samples if len(array_indices) >= num_samples else len(array_indices)
                    print(f"- Numero di Pattern di classe reciprocal_all -> label :{label}, da estrarre: {extracts_samples}")
                    unbalanced_indices.extend(random.sample(array_indices, extracts_samples ))
            


            indices = np.array(unbalanced_indices)
            # aggiornare i target 
            print(Fore.MAGENTA+f"Num tot indice {len(indices)}"+Fore.RESET)
            targets = targets[indices] # aggiorno i targets di dataset all da considerare
            print(Fore.MAGENTA+f"Num tot TARGETS {len(targets)}"+Fore.RESET)



        #-------------------------------------------------
        # split stratificato utilizando gli indici (eventualmente anche aggiornati balanced_indices)
        tmp_indices , test_indices, temp_targets, test_targets  = train_test_split(indices, targets, test_size=test_size, stratify=targets, random_state=42 )

        train_indices, validation_indices, train_targets, validation_targets = train_test_split(tmp_indices, temp_targets, test_size=validation_size/(1-test_size), stratify=temp_targets, random_state=42 )

        # creare i subset dei 3 sets 
        #transforms_train = transforms.Compose([transforms.ToTensor()]) # Qui! utilizzato per il calcolo di mean e devstd!!!!!!!!!!!!
        print("CustomImageFolderSubset: dataset train")
        train = CustomImageFolderSubset(dataset, train_indices, transforms=transforms_train)
        print("QUI SONO IO")
        #train.compute_mean_dev()

        print("CustomImageFolderSubset: dataset validation")
        validation = CustomImageFolderSubset(dataset, validation_indices, transforms=transforms_test)
        test = CustomImageFolderSubset(dataset, test_indices, transforms=transforms_test, extract_sample_for_class= extract_sample_for_class_test) 

        print("Controllo dei targets ")
        print(f"Train :{len(train)}==  {len(train.targets)} and Test: {len(test)} == {len(test.targets)} and Validation: {len(validation)} == {len(validation.targets)}")
        if len(train)== len(train.targets) and len(test) == len(test.targets) and len(validation) == len(validation.targets):
            #if np.array_equal(train_targets,train.targets) and np.array_equal(test_targets , test.targets) and np.array_equal(validation_targets, validation.targets):
                print("Split ok")
        else:
            print(Fore.RED + f"Split doesn't go in success"+ Fore.RESET)
            sys.exit()

        # assegna i vari test alla variabile d'interesse in base al type_dataset 
        if type_task == "known":
            print("QUI SONO IO known")
            self.train_known = train 
            self.validation_known = validation 
            self.test_known = test  
            self.train_Y_known = train_targets 
            self.validation_Y_known = validation_targets 
            #self.train_validation_Y_known = train_targets + validation_targets
            if isinstance(self.train_Y_known, np.ndarray):
                self.train_validation_Y_known = np.concatenate((self.train_Y_known, self.validation_Y_known), axis=0).astype(int).tolist()
                #self.overall_Y_known =  np.concatenate(( self.train_validation_Y_known , self.test_Y_known), axis=0).tolist()
            else:
                self.train_validation_Y_known = self.train_Y_known.extend(self.validation_Y_known)
                #self.overall_Y_known = self.train_validation_Y_known.extend(self.test_Y_known)
            self.test_Y_known = test_targets 
            self.targets_all_known = targets.astype(int).tolist()
            self.overall_Y_known = targets.astype(int).tolist()
            #self.dataset_all_known = dataset[indices]
        
        elif type_task == "reciprocal_class":
            print("QUI SONO IO reciprocal_class")
            self.train_reciprocal = train 
            self.validation_reciprocal = validation 
            self.test_reciprocal = test  
            self.train_Y_reciprocal = train_targets 
            self.validation_Y_reciprocal = validation_targets 
            if isinstance(self.train_Y_known, np.ndarray):
                self.train_validation_Y_reciprocal = np.concatenate((self.train_Y_reciprocal, self.validation_Y_reciprocal), axis=0).astype(int).tolist()
                #self.overall_Y_known =  np.concatenate(( self.train_validation_Y_known , self.test_Y_known), axis=0).tolist()
            else:
                self.train_validation_Y_reciprocal = self.train_Y_reciprocal.extend(self.validation_Y_reciprocal)

            self.test_Y_reciprocal = test_targets 
            self.targets_all_reciprocal = targets.astype(int).tolist()
            self.overall_Y_reciprocal = targets.astype(int).tolist()

            
        

                    
        

    def create_dataset_from_ImageFolder(self, type_task, validation_size, test_size,bool_balance = True, transforms_all=None, transforms_train=None, transforms_test=None, label_mapping=None , num_samples_in_class=None, extract_sample_for_class_test= None, num_samples_for_classes = None):
        

        if type_task == "known":
            # 1) creazione del dataset known (solo con classi desiderate)

            desiderated_classes = self.known_class_name
            folder_root = self.path_folder_photos_known
            label_mapping = label_mapping

            print(Fore.MAGENTA+f"Desiderated classses {desiderated_classes} - {folder_root}"+Fore.RESET) 
        
            #print(Fore.MAGENTA+f"LABEL MAPPING {label_mapping}, desiderated classses {desiderated_classes}"+Fore.RESET) 
            #final_transform = transforms.Compose([*transforms_all.transforms, *transforms_test.transforms])
            dataset = CustomImageFolder(folder_root, desiderated_classes, label_mapping , transforms_all)
            #print(f"target in custom dataset {dataset.targets}" )

            self.known_map_name_to_label = dataset.class_to_idx 
            self.known_map_label_to_name = { label : name for  name , label  in self.known_map_name_to_label.items() } 

            print(f"MAPPA - nome->label {self.known_map_name_to_label}")
            print(f"MAPPA - label->nome {self.known_map_label_to_name}")

            # 2) split stratificato e bilanciato in train/validation/test 
            self.stratified_split_DatasetImageFolder( dataset, type_task, transforms_train, transforms_test, validation_size = validation_size , test_size = test_size, bool_balance = bool_balance, num_el_for_class = None)
    
        elif type_task == "reciprocal_class":
            
            desiderated_reciprocal_classes = self.reciprocal_class_name
            folder_root = self.path_folder_pattern
            label_mapping_reciprocal_class = label_mapping

            print(Fore.MAGENTA+f"Desiderated classses {desiderated_reciprocal_classes} - {folder_root}"+Fore.RESET) 
           
            #print(Fore.MAGENTA+f"LABEL MAPPING {label_mapping}, desiderated classses {desiderated_classes}"+Fore.RESET) 
            #final_transform = transforms.Compose([*transforms_all.transforms, *transforms_test.transforms])
            #transform_all = None #!!!!!!!!!!! blocco qui usato per il calcolo di mean e std
            dataset_reciprocal = CustomImageFolder(folder_root, desiderated_reciprocal_classes, label_mapping_reciprocal_class, transforms_all, transform_in_tensor=True)
            self.reciprocal_map_name_to_label = dataset_reciprocal.class_to_idx 
            self.reciprocal_map_label_to_name = { label : name for  name , label  in self.reciprocal_map_name_to_label.items() } 

            print(f"MAPPA - nome->label {self.reciprocal_map_name_to_label}")
            print(f"MAPPA - label->nome {self.reciprocal_map_label_to_name}")

             # 2) split stratificato e bilanciato in train/validation/test 
            if num_samples_for_classes is not None: 
                num_samples_in_class = num_samples_for_classes
                print( f"num_samples_for_classes - {num_samples_in_class}")

            self.stratified_split_DatasetImageFolder( dataset_reciprocal, type_task, transforms_train, transforms_test, validation_size = validation_size , test_size = test_size, bool_balance = bool_balance, num_el_for_class = num_samples_in_class, extract_sample_for_class_test=extract_sample_for_class_test)

        elif type_task == "unknown":
            desiderated_unknown_classes = self.unknown_class_name 
            folder_root = self.path_folder_photos_unknown 
            label_mapping_unknown_class = label_mapping 
            print(f"Label mapping for unknown {label_mapping_unknown_class}")
            if label_mapping is None:
                label_extra = len(self.known_class_name) + len(self.reciprocal_class_name)
                label_mapping_unknown_class = { idx : label_extra for idx in range(len(desiderated_unknown_classes)) }
                print(Fore.GREEN+f"label_mapping_unknown_class {label_mapping_unknown_class}"+Fore.RESET)

            print(f"extract_sample_for_class_test {extract_sample_for_class_test}")
            extract_sample_for_class_test = extract_sample_for_class_test
            print(f"extract_sample_for_class_test {extract_sample_for_class_test}")
            
            
            print(Fore.MAGENTA+f"Desiderated classses unknown {desiderated_unknown_classes} - {folder_root}"+Fore.RESET) 
            dataset = CustomImageFolder(folder_root, desiderated_unknown_classes, label_mapping_unknown_class, transforms_all, transform_in_tensor=False)
            self.unknown_map_name_to_label = dataset.class_to_idx
            self.unknown_map_label_to_name = { label : name for  name , label  in self.unknown_map_name_to_label.items() } 
            print(f"MAPPA - nome->label {self.unknown_map_name_to_label}")
            print(f"MAPPA - label->nome {self.unknown_map_label_to_name}")

            targets = np.array(dataset.targets)
            indices = np.array([ i for i in range(len(targets)) ]) # array di indici da 0 a lunghezza -1 
            
            dataset_unknown = CustomImageFolderSubset(dataset, indices, transforms=transforms_test, extract_sample_for_class= extract_sample_for_class_test) 
            print("Controllo dei targets ")
            print(f" Dataset Unknown :{len(dataset_unknown)}==  {len(dataset_unknown.targets)}" )
            if len(dataset_unknown)== len(dataset_unknown.targets) :
            #if np.array_equal(train_targets,train.targets) and np.array_equal(test_targets , test.targets) and np.array_equal(validation_targets, validation.targets):
                print("Create Dataset unknown ok")
            else:
                print(Fore.RED + f"Create Dataset Unknwon NO"+ Fore.RESET)
                sys.exit()

            self.test_unknown = dataset_unknown
            self.test_Y_unknown =  dataset_unknown.targets

            print(f"Target unknwon {self.test_Y_unknown[:10]}")
            



    def compute_mean_std(self):
        loader = DataLoader(self.train_known_reciprocal, batch_size=500, shuffle=False, num_workers=2)
        mean = 0.
        std = 0.
        total_images_count = 0

        for images, _ in loader:
            # images shape: (B, C, H, W)
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)  # (B, C, H*W)
            
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images_count += batch_samples

        mean /= total_images_count
        std /= total_images_count

        print(f"Mean: {mean}")
        print(f"Std: {std}")

        sys.exit()


    
    def stratified_split_Dataset(self, dataset, type_task, transforms_train, transforms_test,  validation_size, bool_balance = True, num_el_for_class = None):


        # effettua split stratificato su un dataset solo tra train e validation
        # dataset -> istanza di Dataset o una sua sottoclasse 
        targets = np.array(dataset.targets)
        indices = np.array(dataset.indices) # array di indici da 0 a lunghezza -1 


        # controllare il numero degli elementi per classe
        class_counts = collections.Counter(targets.tolist())
        print(Fore.MAGENTA+f"Num sample for classes {class_counts}"+Fore.RESET)
        min_num_samples = min(class_counts.values()) 
        print(Fore.MAGENTA+f"Min num for class {min_num_samples}"+Fore.RESET)


        # Per ogni classe raccogli gli indici e solo se l'etichetta è fra quelle desiderate (controllo superfluo, visto che dataset già contiene le soli classi d'interesse)
        class_counts = {}
        for idx, label in enumerate(targets):
            if label not in class_counts:
                    class_counts[label] = []
            class_counts[label].append(idx)

        if num_el_for_class: # numero di elementi per classe desiderato 
            if num_el_for_class < min_num_samples: # se è minore rispetto a quello effettivo, prendi quello effettivo
                min_num_samples = num_el_for_class 

        
        if bool_balance == True: 
            balanced_indices = []
            # Seleziona casualmente lo stesso numero di immagini per ogni classe
            for indices in class_counts.values():
                balanced_indices.extend(random.sample(indices, min_num_samples ))
            # aggiungo gli indici 
            indices = np.array(balanced_indices)
            # aggiornare i target 
            print(Fore.MAGENTA+f"Num tot indice {len(indices)}"+Fore.RESET)
            targets = targets[indices] # aggiorno i targets di dataset all da considerare
            print(Fore.MAGENTA+f"Num tot TARGETS {len(targets)}"+Fore.RESET)
        
        #-------------------------------------------------
        # split stratificato utilizando gli indici (eventualmente anche aggiornati balanced_indices)
        #                                                                               test_size=validation_size 
        train_indices , validation_indices, train_targets, validation_targets  = train_test_split(indices, targets, test_size=validation_size, stratify=targets, random_state=42 )
        print("Chiamata Custom Subset su train")
        #transforms_train = transforms.Compose([transforms.ToTensor()]) # qui!!!!!!! blocca le trasformazioni usato per il calcolo di mean e std
        train = CustomSubset(dataset, train_indices, transforms=transforms_train)


        validation = CustomSubset(dataset, validation_indices, transforms=transforms_test)

        print("Controllo dei targets ")
        if len(train_targets)== len(train.targets) and  len(validation_targets) == len(validation.targets):
            if np.array_equal(train_targets,train.targets) and np.array_equal(validation_targets, validation.targets):
                print("Split ok")
        else:
            print(Fore.RED + f"Split doesn't go in success"+ Fore.RESET)
            sys.exit()

        # assegna i vari test alla variabile d'interesse in base al type_dataset 
        if type_task == "known":
            self.train_known = train 
            self.validation_known = validation 
            self.train_Y_known = train_targets 
            self.validation_Y_known = validation_targets 
            self.targets_all_known = targets.astype(int).tolist()




    def create_dataset_from_Datasets(self, root, type_task,type_dataset, validation_size, transforms_all = None, transforms_train = None, transforms_test = None, max_num_sample = None, extract_sample_for_class_test=None):

        label_mapping = None
        if type_task == "known":
            # 1) creazione del dataset known (solo con classi desiderate)
            desiderated_classes = self.known_class_name
            folder_root = self.path_folder_photos_known
            if self.known_assign_place is  None:
                label_mapping = None
            else:
                label_mapping = { int(str_idx):place  for str_idx, place in self.known_assign_place.items()  }
            
            print(f"Assign place KNOWN {label_mapping}")

            # set train-validation
            print(Fore.MAGENTA+f"Desiderated classses {desiderated_classes} - {folder_root} - {label_mapping}"+Fore.RESET)
            dataset_train_tmp  = CustomDatasetKnown(root, desiderated_classes,None, label_mapping , bool_train = True, save_dir= folder_root, type_dataset=type_dataset) 
            #file_csv_of_path = f"{type_dataset}_path_img_train_validation_known.csv"
            #df = pd.DataFrame(dataset_train_tmp.list_path_img)


            # 2) effettture lo split train/validation e creazione dei subset
            transforms_train  = transforms.Compose([*transforms_all.transforms, *transforms_train.transforms])
            transforms_test =  transforms.Compose([*transforms_all.transforms, *transforms_test.transforms])

            self.stratified_split_Dataset(dataset_train_tmp, type_task, transforms_train, transforms_test,  validation_size, bool_balance = True, num_el_for_class = None)

            # 3) creazione del subset del set test 
            dataset_test  = CustomDatasetKnown(root, desiderated_classes,None, label_mapping , bool_train = False, save_dir= folder_root, type_dataset=type_dataset) 
            test_indices = dataset_test.indices
            #file_csv_of_path = f"{type_dataset}_path_img_test_known.csv"
            #df = pd.DataFrame(dataset_test.list_path_img)

            dataset_test = CustomSubset(dataset_test, test_indices, transforms=transforms_test)
            self.test_known = dataset_test
            self.test_Y_known =  dataset_test.targets

            print(f"{len(self.test_known)}== {len(self.test_Y_known)}")

            self.known_map_name_to_label = dataset_train_tmp.class_to_idx 
            self.known_map_label_to_name = dataset_train_tmp.idx_to_class 

            print(f"MAPPA - nome->label {self.known_map_name_to_label}")
            print(f"MAPPA - label->nome {self.known_map_label_to_name}")
            if isinstance(self.train_Y_known, np.ndarray):
                self.train_validation_Y_known = np.concatenate((self.train_Y_known, self.validation_Y_known), axis=0).astype(int).tolist()
                self.overall_Y_known =  np.concatenate(( self.train_validation_Y_known , self.test_Y_known), axis=0).astype(int).tolist()
            else:
                self.train_validation_Y_known = self.train_Y_known.extend(self.validation_Y_known)
                self.overall_Y_known = self.train_validation_Y_known.extend(self.test_Y_known)
        

        
        elif type_task == "unknown":
            print("!!!!!!!!!! Creazione del set di test degli UNKNOWN")
            # 1) creazione del dataset unknown (solo con classi desiderate e numero max di campioni per classe )
            desiderated_classes = self.unknown_class_name
            folder_root = self.path_folder_photos_unknown
            label_extra = len(self.known_class_name) + len(self.reciprocal_class_name)
            replace_label = { idx : label_extra for idx in range(len(desiderated_classes)) }
            print(Fore.GREEN+f"replace_label {replace_label}"+Fore.RESET)
            
            # set train-validation
            transforms_test =  transforms.Compose([*transforms_all.transforms, *transforms_test.transforms])

            print(Fore.MAGENTA+f"Desiderated classses unknown {desiderated_classes} - {folder_root} - num_extract_for_classes {extract_sample_for_class_test}"+Fore.RESET)
            dataset_test_unknown  = CustomDatasetUnknown(root, desiderated_classes,extract_sample_for_class_test, replace_label , bool_train = False, save_dir= folder_root, type_dataset=type_dataset) 
            test_unknown_indices = dataset_test_unknown.indices
            dataset_test_unknown = CustomSubset(dataset_test_unknown, test_unknown_indices, transforms=transforms_test)
            
            self.test_unknown = dataset_test_unknown
            self.test_Y_unknown =  dataset_test_unknown.targets
            print(f"{len(self.test_Y_unknown)} == {len(self.test_unknown)}")

            self.unknown_map_name_to_label = { name : label_extra for name in desiderated_classes}

            print(f"MAPPA - nome->label {self.unknown_map_name_to_label}")
            print(f"{dataset_test_unknown.counter_for_classes()}")

       


    def create_dataset_random_from_Dataset(root, type_task,  type_dataset, transforms_all, transforms_test , num_random_sample):

        if type_dataset == "unknown":
            folder_root = self.path_folder_photos_unknown
            label_extra = len(self.known_class_name) + len(self.reciprocal_class_name)
            replace_label = { 0 : label_extra}
            print(Fore.GREEN+f"replace_label {replace_label}"+Fore.RESET)
            transforms_test =  transforms.Compose([*transforms_all.transforms, *transforms_test.transforms])
            dataset_test_unknown = CustomDatasetRandom(root, num_random_sample, replace_label,bool_train=False, transform=transforms_test, save_dir= folder_root, type_dataset = type_dataset)



    def concat_sets(self, type_dataset):
        if type_dataset == "known_reciprocal":
            train_Y_known_reciprocal  =  np.concatenate([self.train_Y_known ,self.train_Y_reciprocal ])
            validation_Y_known_reciprocal =  np.concatenate([self.validation_Y_known, self.validation_Y_reciprocal])
            test_Y_known_reciprocal  =  np.concatenate([self.test_Y_known,self.test_Y_reciprocal])

            self.train_known_reciprocal  = ShuffledConcatDataset([self.train_known, self.train_reciprocal], train_Y_known_reciprocal)
            self.validation_known_reciprocal  = ShuffledConcatDataset([self.validation_known, self.validation_reciprocal], validation_Y_known_reciprocal)
            self.test_known_reciprocal = ShuffledConcatDataset([self.test_known, self.test_reciprocal], test_Y_known_reciprocal)
            
            self.train_Y_known_reciprocal = self.train_known_reciprocal.get_concat_targets()
            self.validation_Y_known_reciprocal = self.validation_known_reciprocal.get_concat_targets() 
            self.test_Y_known_reciprocal = self.test_known_reciprocal.get_concat_targets()



            print(Fore.CYAN+f"CONCATENATE DATASET KNOWN & RECIPROCAL")
            print(f"LUNGHEZZA FINALE TRAIN {len(self.train_known_reciprocal)} == {len(self.train_Y_known_reciprocal)}")
            print(f"LUNGHEZZA FINALE VALIDATION {len(self.validation_known_reciprocal)} == {len(self.validation_Y_known_reciprocal)}")
            print(f"LUNGHEZZA FINALE TEST {len(self.test_known_reciprocal)} ==  {len(self.test_Y_known_reciprocal)}")

            self.known_reciprocal_map_label_to_name = self.known_map_label_to_name | self.reciprocal_map_label_to_name
            print(f"Map known&reciprocal -> label to name { self.known_reciprocal_map_label_to_name }") 
            self.known_reciprocal_map_name_to_label = {v:k for k, v in self.known_reciprocal_map_label_to_name.items() }
            print(f"Map known&reciprocal -> name to label  {self.known_reciprocal_map_name_to_label}")

        #------------------------------------------------------------------------------------------------------------
        elif type_dataset == "known_reciprocal_unknown":
            # for openn set testing 
            print(f"lUNGHEZZA DATASET DI TEST UNKNOWN {len(self.test_unknown)}")
            test_Y_known_reciprocal_unknown  =  np.concatenate([self.test_Y_known,self.test_Y_reciprocal, self.test_Y_unknown])
            self.test_known_reciprocal_unknown = ShuffledConcatDataset([self.test_known, self.test_reciprocal, self.test_unknown], test_Y_known_reciprocal_unknown)
            self.test_Y_known_reciprocal_unknown = self.test_known_reciprocal_unknown.get_concat_targets()

            print(Fore.CYAN+f"CONCATENATE DATASET KNOWN & RECIPROCAL & UNKNOWN"+Fore.RESET)
            print(f"LUNGHEZZA FINALE TEST {len(self.test_known_reciprocal_unknown)} ==  {len(self.test_Y_known_reciprocal_unknown)}")
            num_classes_K_R = len(self.known_class_name) + len(self.reciprocal_class_name)
            self.known_reciprocal_map_label_to_name = self.known_map_label_to_name | self.reciprocal_map_label_to_name
            self.known_reciprocal_map_label_to_name = { idx:  self.known_reciprocal_map_label_to_name.get(idx) for idx in range(num_classes_K_R)}
            print(f"Map known&reciprocal -> name to label  {self.known_reciprocal_map_label_to_name}")
            self.known_reciprocal_unknown_map_name_to_label = self.known_map_name_to_label | self.reciprocal_map_name_to_label | self.unknown_map_name_to_label
            print(f"Map known&reciprocal&unknown -> name to label  {self.known_reciprocal_unknown_map_name_to_label}")
            print(f"Counter TEST - KNOWN + RECIPROCAL + UNKNOWN { Counter(self.test_Y_known_reciprocal_unknown)}")
