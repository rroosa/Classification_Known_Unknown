import os 
import json 
import sys 
import numpy as np 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import csv
import pandas as pd
import subprocess 
import signal
from colorama import Fore
import random 
import torch
from sklearn.neighbors import LocalOutlierFactor

def create_Folder( path_folder):
	# Create the directory
	try:
		os.mkdir(path_folder)
		print(f"Directory '{path_folder}' created successfully.")
	except FileExistsError:
		print(f"Directory '{path_folder}' already exists.")
	except PermissionError:
		print(f"Permission denied: Unable to create '{path_folder}'.")
	except Exception as e:
		print(f"An error occurred: {e}")

def check_exists(path_file):
    if os.path.exists(path_file):
        #print("File exists")
        return True
    else:
        #print("File doesn't exist")
        return False

def check_file(file_path):
    if os.path.isfile(file_path):
        #print("File exists")
        return True
    else:
        #print("File doesn't exist")
        return False

def check_folder(file_path):
    if os.path.isdir(file_path):
        print("File exists")
        return True
    else:
        print("File doesn't exist")
        return False

def remove_file(path_file):

	if os.path.exists(path_file):
		try:
			os.remove(path_file)
		except Exception as e:
			print(f"Error remove file {path_file}")
	else:
		print("File is not present")



def add_update_key(file_path, key, subkey, value ):

    try:
        with open(file_path) as file:
            data_json = json.load(file)
    except Exception as e:
        print(f"Error load file {e}")
        sys.exit()

    if key not in data_json:
        # add new key in file-json
        if subkey is not None:
            data_json[f"{key}"] = { f"{subkey}": value} 
        else:
            data_json[f"{key}"] = value
    else:
        # key is present
        data_key = data_json[f"{key}"]
        if subkey is not None:
            print(f"Add value under {subkey}")
            data_key[f"{subkey}"] = value # add subkey / or update subkey value
            data_json[f"{key}"] = data_key # update file-json
        else:
            if isinstance(value,dict):
                for sub_key, val in value.items():
                    data_key[sub_key] = val
            else:
                data_json[f"{key}"] = value # sostituisco con un nuovo valore
        
    try:
        with open(file_path, 'w') as file:
            json.dump(data_json, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error write file{e}")
        sys.exit()

def get_key(path_file, subkey, key):
    try:
        with open(path_file, "r") as file:
            data_json = json.load(file)

        value_json = data_json.get(f"{subkey}")
        if value_json is not None and key is not None:
            value = value_json.get(f"{key}")
            return value
        elif value_json is not None and key is None:
            
            return value_json

        else:
            return None 

    except Exception as e:
        print(f"Error load file {e}")
        sys.exit()



def show_sample(sample_image, class_name, label, norm, idx):

    #print(type(sample_image)) # tensor

    # convertire l'immagine in un formato visualizzabile da matplotlib (da tensor a numpy array)
    # e ripristinare la normalizzazione invertendo l'operazione.
    

    if norm is not None:
        # Ripristiniamo la normalizzazione (invertiamo la normalizzazione)
        mean = np.array(norm[0])
        std = np.array(norm[1])
        ####----------------------####   ####----------------------####
        ##  z = (x - mean) / std    ##   ##   x = (z *std ) + mean   ##
        ####----------------------####   ####----------------------####

        sample_image = (sample_image * std[:,None,None] ) + mean[:,None,None]  # Invertiamo la normalizzazione, recupero x
        #è consigliabile denormalizzarla prima della visualizzazione per ottenere una rappresentazione visiva realistica.

    # Limitare i valori dell'immagine tra 0 e 1 per visualizzarla correttamente
    sample_image = np.clip(sample_image, 0, 1)

    sample_image = np.transpose(sample_image,(1, 2, 0))  # Cambiamo l'ordine delle dimensioni (C,H,W) -> (H,W,C)
    
    sample_image = np.array(sample_image * 255, dtype= np.uint8)  # superfluo per matplotlib
    
    # Mostra l'immagine con matplotlib
    plt.imshow(sample_image)
    plt.title(f"Sample:[{idx}], Label: {label}, Class_name: {class_name}")
    plt.axis('off')  # Disabilitare  gli assi
    plt.show()

    #print(matplotlib.get_backend()) # backend di qtagg

def show_image_by_path(image_path, title= None):
    # Caricare e mostrare l'immagine
    plt.ion()
    try:
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis("off")  # Rimuove gli assi per una visualizzazione più pulita
        if title is not None:
            plt.title(title)
        plt.show()
    except Exception as e:
        print("Error")


def extract_string(file_name, prefix, suffix=".json"):
    # Trova la posizione di "config_" e ".json"
    start_index = file_name.find(prefix) + len(prefix)
    end_index = file_name.find(suffix)

    # Estrai la stringa tra "config_" e ".json"
    if start_index != -1 and end_index != -1:
        extracted_string = file_name[start_index:end_index]
        print("Stringa estratta:", extracted_string)
        return extracted_string
    else:
        None


def write_intestation_csv(file_path, reciprocal_pattern, num_classes):

    with open(file_path, mode="w", newline="") as file:
        title = f"Generation of reciprocal patterns at {reciprocal_pattern}"

        header = ["N° run","idx_generation", "idx_pattern", "Fitness"]
        for i in range(num_classes) :
            header.append(f"Prob. {i}")

        header.append("Predicted Label")
        
        writer = csv.writer(file)
        writer.writerow([title])
        writer.writerow(header)

def convert_csv_to_excel(dest):
    df = pd.read_csv(dest, encoding='utf-8')
    print(df)

    dest = dest.replace(".csv", ".xlsx")
    GFG = pd.ExcelWriter(dest)
    # Salviamo il file come Excel
    df.to_excel(GFG, index=False)
    #GFG.save()

def find_num_run(file_path):
    try:
        with open(file_path) as file:
            data_json = json.load(file)
    except Exception as e:
        print(f"Error load file {e}")
        sys.exit()

    if "run" not in data_json:
        return 1 
    else:
        keys = list(data_json["run"].keys())
        key_s = max(keys, key=int)
        run_next = int(key_s) + 1
        return run_next


def write_init_file_json(path, init):
    try:
        with open(path, "w") as file:
            json.dump(init, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error write file{e}")
        sys.exit()


def show_missclassification(path_file_miss, key_exp, miss_class_id_or_all):
    plt.ion()
    print(f"{path_file_miss} - {key_exp} - ")
    if isinstance(miss_class_id_or_all, bool) and miss_class_id_or_all == True:
        obj = get_key(path_file_miss,key_exp, None )
        if obj is not None:

            for id_img, content in obj.items():

                pa = content.get("path_image")
                t = list(content.get("target").keys())[0]
                v = list(content.get("target").values())[0]
                p = list(content.get("predicted_label").keys())[0]
                p2 = list(content.get("predicted_label").values())[0]
                print(f"ID: {id_img} - {pa} - Target: ({t}:{v}) - Predicted Label: ({p}:{p2}) ")

    elif isinstance(miss_class_id_or_all, int):
        
        id_img = miss_class_id_or_all

        obj = get_key(path_file_miss,key_exp, id_img )

        if obj is not None:
            path = obj.get("path_image")
            target  = obj.get("target")
            predicted_label = obj.get("predicted_label")

            if path is not None:
                title = f"Target: ({list(target.keys())[0]} | name class = {list(target.values())[0]}), Predicted Label ({list(predicted_label.keys())[0]} | name class ={list(predicted_label.values())[0]})"
                print(title)
                show_image_by_path(path,title )
        else:
            print(f"Id_img {id_img} is not present")







    elif isinstance(miss_class_id_or_all, str):
        all_img = miss_class_id_or_all


def stop_tensorboard(port):
    # Trova il PID del processo tensorboard che sta ascoltando sulla porta
    try:
        result = subprocess.check_output(f"lsof -t -i:{port}", shell=True)
        pid = result.strip().split(b'\n')  # La 'b' indica che la stringa è in formato bytes)
        pids = [int(line) for line in pid]
        # Termina il processo di TensorBoard
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
            print(f"Processo TensorBoard sulla porta {port} terminato.")
    except subprocess.CalledProcessError:
        print(f"Nessun processo trovato sulla porta {port}.")

def start_tensorboard(logdir_root, port):
    # Controlla se TensorBoard è già in esecuzione e, in caso, termina il processo
    stop_tensorboard(port)
    # Avvia il nuovo processo di TensorBoard
    print(f"Log dir root {logdir_root}")
    command = f"tensorboard --logdir={logdir_root} --port={port} --host=localhost"
    print(f"Avvio TensorBoard con il comando: {command}")  # Debug
    subprocess.Popen(command, shell=True) 

def create_file_json(path, data):
    # Creazione e scrittura su file JSON
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)  # indent=4 

        print(Fore.GREEN+"JSON file created successfully!"+ Fore.RESET)
    except PermissionError:
        print(Fore.RED+"Error: Insufficient permissions to create file"+ Fore.RESET)
    except FileNotFoundError:
        print(Fore.RED+"Error: path is not valid."+ Fore.RESET)
    except OSError as e:
        print(Fore.RED+f"System Error: {e}"+ Fore.RESET)
    except Exception as e:
        print(Fore.RED+f"Error: {e}"+ Fore.RESET)

def create_file_json_known(path):
    data = {
    "project": "known_classes",
    "description": "Training the model on dataset of known classes",
    "folder_photos": "MNIST_phostos | Imagenet_photos",
    "known_name_classes": [],
    "src_dataset": "MNIST | IMAGENET",
    "experiment_[<num>]": {
        "num_epochs": 10,
        "batch_size":0,
        "balanced": "true | false",
        "network": {
            "architecture": "Net | ResNet18",
            "pretrained": "false | true"
        },
        "balanced": "<bool> true | false",
        "hyperparameters": {
            "lr": 0,
            "weight_decay": 0,
            "momentum": 0
        },
        "criterion": {
            "type": "CrossEntropyLoss"
        },
        "scheduler": {
            "type": "<bool> false | StepLR | ReduceLROnPlateau"
        }
    }}

    create_file_json(path, data)

def create_file_json_pattern(path_file_config):
    data = {    
        "description": "Configuration file for pattern generation ",
        "ref_experiment": {
            "number": "experiment_[<>]",
            "src_dataset": "MNIST | IMAGENET",
            "model_type": "Net | ResNet18",
            "architecture_obj": "Net_MNIST | ResNet18_IMAGENET",
            "task_classification": {
                "num_classes": '<>',
                "type": "known"
            },
            "filename_model": "model_experiment_[<>].pth"
        },
        "reciprocal_pattern": [
            {
                "0": "name_class"
            },
            { "1": "name_class" },
            { "2": "name_class"}
        ]
    
    }
    create_file_json(path_file_config, data)

def create_file_json_known_pattern(path_file_config):
    data = {
    "project": "known_pattern_classes",
    "description": "Training the model on dataset of known classes and pattern",
    "folder_photos": "MNIST_photos |  Imagenet_photos",
    "known_name_classes": [ "" ],
    "known_assign_place": {},
    "src_dataset": "MNIST | IMAGENET",
    "folder_pattern_root": "Mnist_patterns | Imagenet_patterns",
    "pattern_experiment": "pattern_experiment_[<>]",
    "reciprocal_name_pattern": [ "" ],
    "reciprocal_assign_place":{},
    "idx_reciprocal_class":[],
    "experiment_[<>]": {
        "num_epochs": 20,
        "batch_size": 64,
        "balanced": "<BOOL> false | true",
        "legend_for_matrix":[],
        "legend_for_plot":{},
        "network": {
            "architecture": "Net | ResNet18",
            "pretrained": "<BOOL> false | true "
        },
        "hyperparameters": {
            "lr": 0,
            "weight_decay": 0,
            "momentum": 0
        },
        "scheduler":{
            "type": "None | StepLR | ReduceLROnPlateau"
        },
        "criterion": {
            "type": "CrossEntropyLoss"
        },
    }}

    create_file_json(path_file_config, data)

def create_file_json_open_set_testing(path_file_config):
    data = {
        "project": "known_unknown_classes",
        "description": "Configuration file for open-set testing",
        "src_dataset_known": "MNIST  | IMAGENET",
        "known_photos_folder": "MNIST_photos | Imagenet_photos",
        "known_classes_name": [],
        "known_assign_place": {},
        "src_dataset_unknown": "MNIST | IMAGENET",
        "unknown_photos_folder": "MNIST_photos | Imagenet_photos",
        "unknown_classes_name":[],
        "reciprocal_pattern_folder": "Mnist_patterns | Imagenet_patterns",
        "pattern_experiment": "pattern_experiment_[<>]",
        "reciprocal_classes_name": [],
        "reciprocal_assign_place":{},
        "idx_reciprocal_class":[],
        "legend_for_plot":{},
        "legend_for_matrix":[],
        "ref_experiment": {
            "exp_num": "experiment_[<>]",
            "src_dataset": "MNIST | IMAGENET",
            "model_type": "Net | ResNet18",
            "architecture_obj": "Net_MNIST | ResNet18_IMAGENET",
            "pretrained": "<BOOL>",
            "balanced": "<BOOL>",
            "batch_size":"<>",
            "task_classification": {
                "num_known_classes": 0,
                "num_reciprocal_classes": 0,
                "type": "known_reciprocal"
            },
            "filename_model": "model_experiment_[<>].pth",
            "root_results_folder": "Results_dataset_known_pattern"
        },
        "task_classification": {
            "type": "open_set_testing",
            "num_known_classes": 0,
            "num_reciprocal_classes": 0,
            "num_unknown_classes":0
        }
    }
    create_file_json(path_file_config, data)









def read_distribution(path_file_distribution_ref_exp, key, subkey):
    print(f"READ distribution from key {key} - subkey {subkey}")
    try:
        with open(path_file_distribution_ref_exp) as file:
            data_json = json.load(file)
    except Exception as e:
        print(f"Error load file {e}")
        sys.exit()
    
    if data_json.get(key):
        info_key = data_json.get(key)
        if info_key.get(subkey):
            subkey = info_key.get(subkey) 
            return subkey 
        else:
            print(Fore.RED+f"Subkey {subkey} is not present"+Fore.RESET)
            return None
    else:
        print(Fore.RED+f"Key {key} is not present"+Fore.RESET)
        return None

def read_list_distance_prototipe(file_config_ref_exp, key, subkey):
    print(f"READ list distance from {key} - subkey {subkey}")
    try:
        with open(file_config_ref_exp) as file:
            data_json = json.load(file)
    except Exception as e:
        print(f"Error load file {e}")
        sys.exit()
    
    if data_json.get(key):
        info_key = data_json.get(key)
        if info_key.get(subkey):
            subkey = info_key.get(subkey) 
            lista = [subkey.get(f"{classe}") for classe in range(len(list(subkey.keys()))) ]
            return lista 
        else:
            print(Fore.RED+f"Subkey {subkey} is not present"+Fore.RESET)
            return None
    else:
        print(Fore.RED+f"Key {key} is not present"+Fore.RESET)
        return None


def read_list_distance_centriode(file_config_ref_exp, key, subkey):
    print(f"READ list distance centriode from {key} - subkey {subkey}")
    # dizionario del tipo { "0": {"vector_centroide":vettore centriode, "distance":distanza} }
    try:
        with open(file_config_ref_exp) as file:
            data_json = json.load(file)
    except Exception as e:
        print(f"Error load file {e}")
        sys.exit()
    lista = []
    if data_json.get(key):
        info_key = data_json.get(key)
        if info_key.get(subkey):
            dict_subkey = info_key.get(subkey)
            num_keys = len(list(dict_subkey.keys())) 
            for keys_int in range(num_keys):
                item = dict_subkey.get(f"{keys_int}")
                lista.append((item.get("vector_centroide"), item.get("distance")))
            return lista 
        else:
            print(Fore.RED+f"Subkey {subkey} is not present"+Fore.RESET)
            return None
    else:
        print(Fore.RED+f"Key {key} is not present"+Fore.RESET)
        return None

def read_dev_std_offset(file_config_ref_exp, key, sub_key):
    print(f"READ  from {key} - subkey {sub_key}")
    # dizionario del tipo { "0": {"vector_centroide":vettore centriode, "distance":distanza} }
    try:
        with open(file_config_ref_exp) as file:
            data_json = json.load(file)
    except Exception as e:
        print(f"Error load file {e}")
        sys.exit()
    
    if data_json.get(key):
        info_key = data_json.get(key)
        if info_key.get(sub_key):
            value = info_key.get(sub_key)
            return value 
        else:
            return None 
    else:
        return None




def list_min_prob_for_classes(read_distribution):
    min_prob_for_class = []
    num_classes = len(read_distribution.keys())
    #print(f"num_claase {num_classes}")

    for i in range(num_classes):
        obj_class = read_distribution.get(f"class_{str(i)}")
        class_id = str(i)
        print(f"{i} {obj_class} ")
        prob_obj_of_class = obj_class.get(f"{class_id}")
        prob_min = prob_obj_of_class.get("min") 
        min_prob_for_class.append(prob_min)
    
    #print(min_prob_for_class)
    return min_prob_for_class

def list_max_prob_for_classes_of_unknown(read_distribution):
    max_prob_for_class = []
    index_unknow = len(read_distribution.keys()) -1
    print(f"num_claase {index_unknow}")

    obj_class = read_distribution.get(f"class_{str(index_unknow)}")
    for k, min_max in obj_class.items():
        max_v = min_max.get("max")  
        print(f"{k}: max {max_v} ")
        max_prob_for_class.append(max_v)
    
    print(max_prob_for_class)
    return max_prob_for_class


    

def  select_random_name_classes_Imagenet_unknown(num_ele, path ):
    print(path)
    try:
        with open(path, "r") as file_json:
            data_json = json.load(file_json)
            values_list = data_json.values() # lista si liste [ [id, nome], [id,nome], [], ...]

            list_name = [ lista[1] for lista in values_list]
            selected_name = random.sample(list_name, num_ele)
            print(f"Selected name are {selected_name}")
            return selected_name


    except Exception as e:
        print(f"Error load file {e}")
        sys.exit()


def lunch_subprocess(*args):
    command = []
    command.append("python3")
    command.extend(list(args))
    print(Fore.MAGENTA+f"Command: {command}"+Fore.RESET)
    subprocess.run(command)

def plot_fitness(stats,path_fold,filename,  title="Fitness over Generations"):
    path_fold = os.path.join(path_fold, filename)
    generations = range(len(stats.most_fit_genomes))
    best = [g.fitness for g in stats.most_fit_genomes]
    mean = stats.get_fitness_mean()
    stdev = stats.get_fitness_stdev()

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best, label="Miglior fitness", color="blue")
    plt.plot(generations, mean, label="Fitness media", color="green")
    plt.fill_between(generations, 
                     [m - s for m, s in zip(mean, stdev)],
                     [m + s for m, s in zip(mean, stdev)],
                     alpha=0.2, color="green", label="Deviazione standard")
    plt.title(title)
    plt.xlabel("Generazione")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_fold)
    plt.show()

def plot_genome_complexity(stats, path_fold, filename):
    path_fold = os.path.join(path_fold, filename)
    generations = range(len(stats.most_fit_genomes))
    num_nodes = [len(g.nodes) for g in stats.most_fit_genomes]
    num_connections = [len(g.connections) for g in stats.most_fit_genomes]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, num_nodes, label="Nodi", color="purple")
    plt.plot(generations, num_connections, label="Connessioni", color="orange")
    plt.xlabel("Generazione")
    plt.ylabel("Numero")
    plt.title("Complessità del genoma (nodi e connessioni)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path_fold)
    plt.show()




def boxplot_outliers_by_class(data_Nd, target_array):
    outlier_indices = []
    #if isinstance(data_3d, torch.Tensor):
    #    data_3d= data_3d.detach().cpu().numpy()
    #else:
    #    data_3d = np.array(data_3d)
    #data_3d = data_3d.cpu().numpy()
    #data_3d = np.array(data_3d)
    print(np.unique(target_array))
    print(f"Shape data_Nd {data_Nd.shape}")
    print(f"Shape target {target_array.shape}")
    #sys.exit()

    for cls in np.unique(target_array):
        # Ottieni gli indici dei campioni della classe corrente
        cls_indices = np.where(target_array == cls)[0] 
        print(type(cls_indices), type(cls_indices[0]))

        print("cls_indices dtype:", cls_indices.dtype)
        #cls_indices = cls_indices.astype(int)
        cls_points = data_Nd[cls_indices]  # shape (n_class_samples, 3)
        #sys.exit()

        # Inizializza maschera per outlier locali alla classe
        outlier_mask = np.zeros(len(cls_indices), dtype=bool)

        # Applica la regola del boxplot su ogni dimensione ()
        N = data_Nd.shape[1]
        print(f"Dimensione N {N}")
        for dim in range(N):
            values = cls_points[:, dim]
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier_mask |= (values < lower) | (values > upper)

        # Aggiungi gli indici globali dei punti outlier
        outlier_indices.extend(cls_indices[outlier_mask])

    return np.array(outlier_indices)



def compute_distribution(label_real, label_pred, prob_output, num_classes, path_distribution, outlier_idx = None):
    print(f"Num classses {num_classes}")
    
    data_ditribution = []
    columns = ["ID_CLASS"]
    column_probs = [f"prob_{i}" for i in range(num_classes)] # chiavi per il dict
    columns.extend(column_probs)
    count_tot = 0
    count_process = 0
    mask = label_pred == label_real
    #print(f" indici di outliner {outlier_idx}+ {len(outlier_idx)}")
    for idx, v in enumerate(mask):
    #print(idx)
        if v == True:
            if outlier_idx is None or (outlier_idx is not None and idx not in outlier_idx):
                class_real = label_real[idx]
                prob_classes = prob_output[idx] # valori
                #print(f"prob_output {prob_output.shape}")
                #print(f"column_probs {len(column_probs)}")
                #print(f"prob_class {prob_classes.shape}")
                #print(f"type prob_class {type(prob_classes)}")
                #print(f"type column_probs {type(column_probs)}")

                # crea un dizionari o partire da lista chiavi e lista valori
                dict_obj = dict(zip(column_probs, list(prob_classes)))
                
                dict_obj["ID_CLASS"] = class_real
                # aggiungere la chiave ID_CLASS : 0

                # aggiungi al dataframe la riga 
                data_ditribution.append(dict_obj)
                count_process =  count_process +1
            
            count_tot = count_tot +1 

    
    # creazione datframe dal dizionario
    df_distribution = pd.DataFrame(data_ditribution)
    df_distribution = df_distribution[columns]
    # salvare come file csv 
    print(Fore.GREEN+f"Save file csv that describes distribution of classes {path_distribution}"+Fore.RESET)
    df_distribution.to_csv(path_distribution, index=False)
    print(df_distribution)
    print(f"!!!!!!!!!!!!!!Num sample pred=target {count_tot}")
    print(f"Num sample processati {count_process}")



def localOutlierFactor(probs, labels):
    print(f"Dimensione probs {probs.ndim}, shape probs {probs.shape}")
    print(f"Dimensione labels {labels.ndim}, shape labels {labels.shape}")
    outlier_indices = []

    for cls in torch.unique(labels):
        # Trova gli indici globali dei campioni di questa classe
        class_idx = (labels == cls).nonzero(as_tuple=True)[0]  # tensor di indici globali
        class_probs = probs[class_idx]

        if class_probs.ndim == 1 or (class_probs.ndim == 2 and len(class_probs < 5)):
            continue

        lof = LocalOutlierFactor(n_neighbors=200)
        print(class_probs.ndim)
        print(class_probs.shape)
        print("Qui",len(class_idx), len(class_probs))
        if class_probs.ndim == 1:
            class_probs = class_probs.reshape(-1, 1)
            # quando c'è solo un campione
        
        print("Qui2",len(class_idx), len(class_probs), class_probs.ndim)

        
        preds = lof.fit_predict(class_probs)  # -1 = outlier

        print("Qui 3",  preds.shape)

        # Ottieni indici outlier all'interno della classe
        class_outlier_idx = class_idx[preds == -1]

        # Aggiungi agli indici totali
        outlier_indices.extend(class_outlier_idx.tolist())
       


    print(f"Totale outlier trovati: {len(outlier_indices)}")
    print("Esempio di indici:", outlier_indices[:10])
    return outlier_indices


def define_min_max(path_distribution, num_classes):
    try:
            df_distribution = pd.read_csv(path_distribution)
            distribution_report = { }
            for class_id in range(num_classes):

                df_class = df_distribution[ df_distribution["ID_CLASS"] == class_id ]

                dict_prob = {}
                for prob_id in range(num_classes):
                    min_prob = df_class[f"prob_{prob_id}"].min() 
                    max_prob = df_class[f"prob_{prob_id}"].max()
                    dict_prob[f"{prob_id}"] = {"min": min_prob, "max": max_prob}

                distribution_report[f"class_{class_id}"] = dict_prob 

            return distribution_report

            

    except FileNotFoundError:
        print(Fore.RED+f"Error file is not exist"+Fore.RESET)
        sys.exit()
    except pd.errors.ParserError:
        print(Fore.RED+f"Error, the format is not valid"+Fore.RESET)
        sys.exit()

    except pd.errors.EmptyDataError:
        print(Fore.RED+f"Error file is empty"+Fore.RESET)
        sys.exit()
    except Exception as e:
        print(Fore.RED+f"Error {e}"+Fore.RESET)
        sys.exit()
    
