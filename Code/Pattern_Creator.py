import torch
import numpy as np 
import json
import os
from functools import partial
import neat
from PIL import Image
import torch.nn.functional as F
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
from PIL import Image
import csv
from torchvision import  transforms 
from torch.utils.tensorboard import SummaryWriter
import math
from meters_fitness import *
import sys
import hashlib
import pickle
import random
from utility import create_Folder, plot_fitness, plot_genome_complexity
import pandas as pd
import copy
import torch
import torch.nn.functional as F

class Pattern_Creator():

    def __init__(self):
        
        self.path_home_dataset = None
        self.path_csv_filename = None
        self.reciprocal_pattern = None
        self.path_model = None
        
        self.model = None

        self.best_patterns_for_generation = {}
        self.num_current_generation = 0
        self.counter_best = 0

        self.compare_fitness = []
        
        self.reciprocal_class = None 
        self.reciprocal_class_name = None

        self.all_reciprocal_pattern =  None
    

        self.file_report_neat = None

        self.device = None

        self.stats = None

        self.population = None

        self.mean = None 
        self.std = None
        self.transformer = None

        self.logdir_root = None

        self.type_generation = None 
        self.bool_RGB = False

        self.best_genoma_id = [] #  # collaziona gli id dei genomi che man mano vengono raccolti
       
        self.previous_genomes = set()

        self.add_bool_transform = True
        self.function_fitness = ""
        self.constraint_name = ""

        self.dest = None 
        self.current_run = None
        self.constrain = None
        self.folder_dest = None
        self.model_best_genoma = None
        self.config_path = ""

        self.input_node = 2
        self.path_folder_class = None
        self.sorgente_folder = None
        self.num_species = None
        self.num_classes_tot = None
        self.all_reciprocal_class = None
        self.all_reciprocal_class_number = None
        self.retrieval_class = None

        self.path_folder_experiment = None
        self.list_min_distribution = None
        self.brother_reciprocal = False

    def set_num_run(self, num_run):
        self.num_run = num_run
    
    def set_folder_config(self, path):
        self.folder_dest = path

    def set_path_csv_filename(self, path):
        self.path_csv_filename = path
    
    def set_path_home_datasets(self, path):
        self.path_home_dataset = path

    
    def save_best_model_genome(self, path_dest, num_run, i):
        if self.model_best_genoma is not None:
            path_dest = os.path.join(path_dest, f"reciprocal_[{self.reciprocal_class}]" )
            create_Folder(path_dest)
            filename = f"run_[{num_run}]_best_[{i}]_[{self.model_best_genoma.fitness:.5f}].pkl"
            path = os.path.join(path_dest, filename)
            print(Fore.MAGENTA+f"Save finded genome"+Fore.RESET)
            with open(path, "wb") as f:
                pickle.dump(self.model_best_genoma, f)


    
    def random_color_jitter(self):
        return transforms.ColorJitter(
            brightness=random.uniform(0, 0.5),  
            contrast=random.uniform(0, 0.5),    
            saturation=random.uniform(0, 0.5),  
            hue=random.uniform(0, 0.2)         
        )




    def add_transforms_rgb(self, pattern):
        #print(f" type pattern {type(pattern)}")
        
        tr  = transforms.Compose([
                transforms.ToPILImage(), # 2) riconverto da Tensor o numpy (h,w,c) a PIL  (serve per applicare le trasformazioni )
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomApply([self.random_color_jitter()], p=0.8),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),#aggiunto
                #transforms.RandomVerticalFlip(p=0.3), # aggiunto
                #transforms.RandomSolarize(threshold=random.randint(64, 192), p=0.3),
                #transforms.RandomEqualize(p=0.5),
                #transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0)),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5)),
                #transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                #transforms.ElasticTransform(alpha=50.0), # torchvision >= 0.13
                transforms.ToTensor(), # 2) riconverto da PIL  Tensor (questo fa si che i valori siano riportati ell'intervallio [0-1])
                #transforms.Lambda(lambda img: img + 0.05 * torch.randn_like(img)),  # Rumore gaussiano
            ])

        pattern_tr = tr(pattern)
        return pattern_tr

        

    def add_transforms_gs(sef, pattern):
        tr  = transforms.Compose([
                transforms.ToPILImage(), # 2) riconverto da Tensor o numpy (h,w,c) a PIL  (serve per applicare le trasformazioni )
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ToTensor(), # 2) riconverto da PIL  Tensor (questo fa si che i valori siano riportati ell'intervallio [0-1])
            ])
        pattern_tr = tr(pattern)
        return pattern_tr

    def get_reciprocal_pattern(self):
        return self.reciprocal_pattern

    def set_net_architecture(self, net_architecture,size_img, path_model, path_best_genoma_model):
        self.model = net_architecture
        self.path_model = path_model
        self.size_input_img = size_img # (3,224,224)
        self.path_best_genoma_model = path_best_genoma_model
        self.load_model()

    def set_filename_report(self,  filename):

        self.file_report_neat = filename
    
    def set_path_folder_experiment(self, path_folder_exp):
        self.path_folder_experiment = path_folder_exp

    def set_normalize(self, inf_norm):
        self.mean = np.array(inf_norm[0])
        self.std = np.array(inf_norm[1])
    
    def set_transformer(self, transformer):
        self.transformer = transformer

    def set_logdir(self,logdir_root):

        self.logdir_root = logdir_root
        os.makedirs(self.logdir_root, exist_ok = True)
        self.writer = SummaryWriter(self.logdir_root)
    
    def set_constrain(self, constrain):
        self.constrain = constrain

    def set_bool_RGB(self, bool_RGB):
        self.bool_RGB = bool_RGB
    
    def set_num_classes_tot(self, num_classes_tot):
        self.num_classes_tot = num_classes_tot
    



    def load_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        checkpoint = torch.load(self.path_model, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
    
    def generate_pattern_RGB(self, net):
        self.type_generation = "spazial_pixel_RGB"
        
        # --- GENERAZIONE DI UN'IMMAGINE USANDO LA RETE NEAT 
        img = np.zeros((self.size_input_img[1], self.size_input_img[2], 3), dtype=np.uint8)  # Immagine vuota per contenere valori unint8 = [0-255]
        for y in range(self.size_input_img[1]):
            for x in range(self.size_input_img[2]):
                # Normalizza le coordinate da [0, img_size] a [-1, 1]
                norm_x = 2 * x / (self.size_input_img[2] - 1 ) - 1
                norm_y = 2 * y / (self.size_input_img[1] - 1 ) - 1
                # Passa le coordinate nella rete NEAT
                output = net.activate([norm_y, norm_x]) # input coordinate normalizzate tra [-1,1]

                """ utilizzare se la funzione di attivazione è tanh
                r = np.clip((output[0] + 1) * 0.5 * 255, 0, 255)
                g = np.clip((output[1] + 1) * 0.5 * 255, 0, 255)
                b = np.clip((output[2] + 1) * 0.5 * 255, 0, 255)
                """
                

                #""" da utilizzare quando la funzione di attivazione è sigmoid
                r = int(np.clip(output[0] * 255, 0, 255))
                g = int(np.clip(output[1] * 255, 0, 255))
                b = int(np.clip(output[2] * 255, 0, 255))
                #"""
                img[y, x] = [int(r), int(g), int(b)]

        return img
    


    def generate_pattern_GS(self, net):
        self.type_generation = "spazial_pixel_GrayScale"

        self.config.genome_config.num_outputs = 1

        img = np.zeros((self.size_input_img[1], self.size_input_img[2],self.size_input_img[0]), dtype=np.uint8)  # Immagine vuota per contenere valori unint8 = [0-255]
        for y in range(self.size_input_img[1]):
            for x in range(self.size_input_img[2]):
                # Normalizza le coordinate da [0, img_size] a [-1, 1]
                norm_x = 2 * x / (self.size_input_img[2] - 1 ) - 1
                norm_y = 2 * y / (self.size_input_img[1] - 1 ) - 1

                
                output = net.activate([norm_y, norm_x]) # input coordinate normalizzate tra [-1,1]
                """ da utilizzare se la funzione di attivazione è tahn 
                value = np.clip((output[0] + 1) * 0.5 * 255, 0, 255)  #se viene usato tanh 
                """

                """ da utilizzare quando la funzione di attivazione è sigmoid"""
                value = int(np.clip(output[0] * 255, 0, 255))
                img[y, x] = value
        
        return img
    

    def evaluate_fitness(self, genomes,config):
         
        best_pattern = None
        best_confidence = 1.0
        fitness = 0.0
        """
        #################### spostato fuori in modo da usare la stessa immagini per tutte le generazioni################ 
        if self.input_node == 5 or self.input_node == 3:
            size = (self.size_input_img[1], self.size_input_img[2])
            print(f"Size {size}")
            self.numpy_img_ref = self.obtain_img_ref(size)
        ###########################################################
        """
        counter_pattern_in_current_generation = 0
        print(Fore.BLUE+ f"!!!!!!!!!!!!!Current generation: {self.num_current_generation} !!!!!!!!!!!!!!!!!!!!!"+Fore.RESET)


        for genome_id, genome in genomes:
            print(Fore.MAGENTA+f"Generation {self.num_current_generation} - Genoma: {genome_id}"+Fore.RESET)
            if self.input_node == 5 or self.input_node == 3:
                size = (self.size_input_img[1], self.size_input_img[2])
                print(f"Size {size}")
                #self.numpy_img_ref = self.obtain_img_ref(size)


            net = neat.nn.FeedForwardNetwork.create(genome, config)

            if self.bool_RGB == True:
                print("Create pattern RGB")
                pattern = self.generate_pattern_RGB(net) # (224,224,3)
                print("shape PATTERN",pattern.shape)
                print(f"shape  INPUT {self.size_input_img}")
            else: # grey scale

                pattern = self.generate_pattern_GS(net) 

                print("shape PATTERN",pattern.shape)
                print(f"shape  INPUT {self.size_input_img}")
                

            tr_tensor = transforms.ToTensor() 
            pattern_tensor = tr_tensor(pattern)

            
            if self.add_bool_transform == True: 
                print("add_bool_transform")
                if self.bool_RGB == True:
                    pattern_tensor = self.add_transforms_rgb(pattern) 
                else:
                    pattern_tensor = self.add_transforms_gs(pattern) 


                print("pattern_with_transform tensor float32",type(pattern_tensor), pattern_tensor.shape)


                tr_pil = transforms.ToPILImage()
                pattern_pil = tr_pil(pattern_tensor)
                pattern = np.array(pattern_pil) 
                print(f"pattern to be save, after applied PIL {type(pattern)}, {pattern.shape}, {pattern.dtype}")
            


            pattern_normalized = self.transformer(pattern) 
      
            pattern_batch = pattern_normalized.unsqueeze(0) # trasform in batch
            pattern_input = pattern_batch.to(self.device)

            # process pattern to model
            with torch.no_grad():
                output = self.model(pattern_input)

            output = F.softmax(output, dim=1)

            num_classes = output.shape[1]

            output_prob = np.asarray(output.to('cpu')[0])
            print(type(output_prob),output.shape, output_prob) # shape 
            predicted_label   = output.to('cpu').max(1)[1].item() 
            # Find  maxiumum value along dim 1 , returns ( value, index ) so take index in position 1 
            


            if self.all_reciprocal_pattern == True:
                

                
                fitness_finale = compute_fitness(output_prob)
                self.function_fitness = "fitness_jsd" 
                
                print(f"Fitness Score {fitness_finale}")
                fitness = fitness_finale
                print(f"Threshold: { self.THRESHOLD_CONFIDENCE}" ) 

                genome.fitness = fitness
             

                if genome.fitness >= self.THRESHOLD_CONFIDENCE: 
                     # save pattern under key = num_generation
                    print(Fore.GREEN+f"Find pattern"+Fore.RESET)
                    c_p = counter_pattern_in_current_generation
                    bool_constraint = True
                    folder = "all"
                    

                    if bool_constraint:
                        self.best_genoma_id.append(genome_id)
                        print(Fore.GREEN+f"Find pattern, genoma_num: {genome_id}, FITNESS {genome.fitness}"+Fore.RESET)

                        if c_p == 0: # if it is first pattern of the current generation     
                            print(Fore.GREEN+f"insert in dict, as first element of the generation"+Fore.RESET)
                            self.best_patterns_for_generation[f"gen_[{self.num_current_generation}]"] = { f"{str(c_p)}":[pattern, genome.fitness ,output.tolist()[0] , predicted_label ]}
                            counter_pattern_in_current_generation = c_p + 1
                            self.counter_best += 1

                        else:
                            print(Fore.GREEN+f"insert in dict as next element of the generation"+Fore.RESET)                                 
                            patterns_obj = self.best_patterns_for_generation[f"gen_[{self.num_current_generation}]"]
                            patterns_obj[f"{str(c_p)}"] = [pattern, genome.fitness , output.tolist()[0], predicted_label]
                            
                            self.best_patterns_for_generation[f"gen_[{self.num_current_generation}]"] = patterns_obj
                            counter_pattern_in_current_generation = c_p + 1
                            self.counter_best += 1 
                        
                        self.model_best_genoma = genome

                        self.save_best_model_genome( self.folder_dest, self.num_run, self.counter_best)

                        #data = [f"{str(genome_id)}", output_prob[0], output_prob[1],  output_prob[2], output_prob[3],genome.fitness, fitness_jsd_one_class,fitness_jsd_one_class_with_penality_min_diff, fitness_jsd_one_class_with_penality_mse,self.THRESHOLD_CONFIDENCE ]
              
        
            if genome.fitness is  None:
                
                genome.fitness = 0.0

        

        self.save_best_pattern_from_numpy( self.dest, self.current_run)
        self.save_report(self.current_run) 

        self.num_current_generation +=1
        self.best_patterns_for_generation = {}


        
        # Temporary patch: ensure that each species contains only members with valid fitness
        for sid, specie in self.population.species.species.items():
            fitnesses = [m.fitness for m in specie.members.values()]
            
            if len(fitnesses) == 0:
                print(f"[ERRORE] Specie {sid} vuota!")
                continue

            # Se qualche fitness è None, rimpiazziamola con 0.0
            for m in specie.members.values():
                if m.fitness is None:
                    print(f"[ERRORE] Genoma {m.key} in specie {sid} ha fitness None, sostituito con 0.0")
                    m.fitness = 0.0

        




    def save_id_genoma(self):
        print(f"List genoma id find TOT= {len(self.best_genoma_id)}")
        print(self.best_genoma_id)



    # generate reciprocal pattern 
    def run_generative_reciprocal_patterns(self, path_prefix, num_run, all_reciprocal_pattern=False,  best_genome = None): 
        self.dest = path_prefix
        self.current_run = num_run
        print(f"All Reciprocal patter: {all_reciprocal_pattern}")
        if all_reciprocal_pattern == True:
            self.all_reciprocal_pattern = True 

            
        self.config_path =  ""
        if self.bool_RGB == True:

            if self.all_reciprocal_pattern:

                self.config_path = "config-feedforward-rgb" 
                
                self.input_node = 2
                self.THRESHOLD_CONFIDENCE = 0.95
                print(f"Config path {self.config_path} ")
                
        else:
            # grey scale
            if self.all_reciprocal_pattern:
                
                self.config_path  = "config-feedforward-gs"
                self.input_node = 2
                self.THRESHOLD_CONFIDENCE = 0.98




        print(f"COnfiguration file NEAT {self.config_path}")
        
        config = neat.Config(neat.DefaultGenome, 
                             neat.DefaultReproduction, 
                             neat.DefaultSpeciesSet, 
                             neat.DefaultStagnation, 
                             self.config_path)

        self.population = neat.Population(config)
        num_generation = 200


        

        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)
        self.num_species = config.pop_size

        self.config = config


        


        if self.all_reciprocal_pattern is not None:
            header = ["ID_GENOMA"]
            for i in range(self.num_classes_tot):
                header.append(f"Prob_{i}")
            header.append("fitness_jsd_minDiff")
            header.append("threshold")

    

        best_genoma = self.population.run(self.evaluate_fitness, 1)

        print(f"BEST genoma - fitness {best_genoma.fitness}")


 
    def save_best_pattern_from_numpy(self, dest, num_run):
            print(Fore.MAGENTA+f"Current Generation {self.num_current_generation}"+Fore.RESET)
            # pattern numpy array type unin8 (H,W,C), without scaling gaussian, 
            for generation, obj in self.best_patterns_for_generation.items():
                for num_pattern, lista in obj.items():
                    pattern = lista[0]
                    fitness = lista[1]
                    
                    print(f"{type(pattern)},{ pattern.shape}, fitness: {fitness}") # è un numpy normalizzato

                    img_name = f"run_[{num_run}]_{generation}_idx_[{num_pattern}]_f_[{fitness:.5f}].jpeg"

                    # Crea un'immagine con PIL
                    image_pil = Image.fromarray(pattern)
                    

                    file_path = os.path.join(dest, img_name)
                    
                    # Save image file .jpeg
                    print(Fore.GREEN+f"Save image {num_pattern}"+Fore.RESET)
                    print(f"SHAPE PATTERN to save  {pattern.shape}")
                    print(f"SHAPE IMAGE to save {image_pil.size}")
                    print(f"mode IMAGE to save  {image_pil.mode}")

                    image_pil.save(file_path, format='JPEG')





    def save_report(self, num_run):

        with open(self.file_report_neat, "a", newline="") as file:
            writer = csv.writer(file)

            for generation, obj in self.best_patterns_for_generation.items():
                for num_pattern, lista in obj.items():
                    #pattern = lista[0]
                    fitness = lista[1]
                    output_prob = lista[2]
                    predicted_label = lista[3]
                    idx_gen = generation.split("gen_[")[1].split("]")[0]
                    lista = [num_run,idx_gen, num_pattern, fitness]
                    for p in output_prob:
                        lista.append(p)
                    lista.append(predicted_label)
                    writer.writerow(lista)

    def info_generation(self):
        info = {"tot_generation": self.num_current_generation,
                "size_popolation": self.num_species,
                "find_best": self.counter_best,
                "th_confidence": self.THRESHOLD_CONFIDENCE,
                "activation_default" : self.config.genome_config.activation_default,
                "activation_options": self.config.genome_config.activation_options,
                "input": self.config.genome_config.num_inputs,
                "generate_pattern":self.type_generation,
                "add_transforms":self.add_bool_transform,
                "function_fitness": self.function_fitness,
                "config_file": self.config_path 
        }
        return info


 

    def run_generate_with_best_genoma(self, path_prefix, num_run, path_model_best,filename_genoma_best, reciprocal_pattern=None, all_reciprocal_pattern=False):
        winner_genoma = None
        

        
        print(f"type  of winner genoma {type(winner_genoma)}")
        
        self.dest = path_prefix
        self.current_run = num_run


        if all_reciprocal_pattern == True:
            self.all_reciprocal_pattern = True 
            config_all = ["config-all-v1", "config-feedforward-imgnet-2"]
            self.config_path = config_all[1]
            self.input_node = 2
            self.THRESHOLD_CONFIDENCE = 0.98
            path_model_best = os.path.join(path_model_best,f"reciprocal_[all]")
            path_model_best = os.path.join(path_model_best,filename_genoma_best)
            print(f"Path of best genoma {path_model_best}")

            try: 
                with open(path_model_best, "rb") as f:
                    winner_genoma = pickle.load(f)
            except Exception as e:
                print(f"Error {e}")
                sys.exit()
        
        
        if self.bool_RGB == True:
  
            if self.all_reciprocal_pattern:
                self.config_path = "config-feedforward-rgb"
                self.input_node = 2
                self.THRESHOLD_CONFIDENCE = 0.95

        else:
            #------ SCALE GREY --------------------
            if self.all_reciprocal_pattern:
                self.config_path  = 'config-feedforward-gs'
                self.THRESHOLD_CONFIDENCE = 0.98
                self.input_node = 2


        config = neat.Config(neat.DefaultGenome, 
                             neat.DefaultReproduction, 
                             neat.DefaultSpeciesSet, 
                             neat.DefaultStagnation, 
                             self.config_path)

        self.population = neat.Population(config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)

        self.num_species = config.pop_size

        self.config = config

        #self.config.reproduction_config.crossover_prob = 0.8
        
        net = neat.nn.FeedForwardNetwork.create(winner_genoma, config)
        
        self.num_current_iteration = 0
        range_num = 1000
        for i in range(range_num):
            print(Fore.MAGENTA+f"Iterazione {i}/{range_num-1}"+Fore.RESET)
            if self.bool_RGB == True:
                pattern = self.generate_pattern_RGB(net) # (224,224,3)
            
            else:
                pattern = self.generate_pattern_GS(net) #
           
            print("DIMENSIONE PATTERN",pattern.shape)
            print(f"DIMENSIONE IN INPUT {self.size_input_img}")
                

            tr_tensor = transforms.ToTensor() 
            pattern_tensor = tr_tensor(pattern)
 

            if self.add_bool_transform == True: 
                print("add_bool_transform", self.add_bool_transform)
                if self.bool_RGB == True:
                    print("bool_RGB", self.bool_RGB)
                    pattern_tensor = self.add_transforms_rgb(pattern) 
                else:
                    pattern_tensor = self.add_transforms_gs(pattern) 
          

            tr_pil = transforms.ToPILImage()
            pattern_pil = tr_pil(pattern_tensor)
            pattern = np.array(pattern_pil) # questo è da salvare uint8[0-255] [h,w,c]
            print(f"pattern da salvare, dopo aver applicato PIL {type(pattern)}, {pattern.shape}, {pattern.dtype}")
            


            pattern_normalized = self.transformer(pattern) 

            ########---------------------------------------------------------------------------------
            #           BATCH 
            # ##################       
            pattern_batch = pattern_normalized.unsqueeze(0) # renderlo batch
            pattern_input = pattern_batch.to(self.device)


            with torch.no_grad():
                output = self.model(pattern_input)

            output = F.softmax(output, dim=1)

            num_classes = output.shape[1]

            output_prob = np.asarray(output.to('cpu')[0])
            print(type(output_prob),output.shape, output_prob) # shape [1,4]
            predicted_label   = output.to('cpu').max(1)[1].item() # Trova il valore massimo lungo la dimensione 1 (cioè fra 4 elementi), restituisce ( valore, indice ) quindi prendiamo l'indice (posizione [1]) 


            if self.all_reciprocal_pattern == True:
               
                fitness= compute_fitness(output_prob)
                self.function_fitness = "fitness_jsd"  
                if fitness >= self.THRESHOLD_CONFIDENCE: 
                        
                    print(Fore.GREEN+f"Find pattern"+Fore.RESET)
                   
                    self.best_patterns_for_generation[f"gen_[{self.num_current_iteration }]"] = { f"{str(i)}":[pattern, fitness ,output.tolist()[0] , predicted_label ]}      
                    self.counter_best += 1
                    self.best_genoma_id.append(i)


            self.save_best_pattern_from_numpy( self.dest, self.current_run)
            self.save_report(self.current_run) 

            self.num_current_iteration +=1
            self.best_patterns_for_generation = {}

    


