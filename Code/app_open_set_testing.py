import torch 
import numpy
import os
import json
import sys
import argparse
from Transformer import * 
from Dataset_Creator import *
from Manager_Networks import *
from utility import *
from training import run_training
from testing import run_testing
from visualizing import run_visualizing
import configparser
from colorama import Fore
from loss_functions import *
#


root = os.getcwd()
folder_result = "Results_open_set_testing"
path_home_dataset = None
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

num_tot_classes = None
num_known_classes = None
num_reciprocal_classes = None
num_unknown_classes = None
num_classes_K_R = None

def set_seed(seed):
    print("Chiamata SEED")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.set_device(device)  
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device


seed = 42 

if __name__ == '__main__':
    device = set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    parser = argparse.ArgumentParser(description='Open-set testing')
    parser.add_argument('--experiment', type=int, choices={1,2}, required= True) 
    parser.add_argument('--config_file', type=str, default= "config.json")
    parser.add_argument('--phase', type=str, choices = {'testing', "visualization", "miss_classification"}, required= True)

    # mutually exclusive group
    group = parser.add_mutually_exclusive_group(required= True)
    group.add_argument('--distribution', action="store_true", help="Collect distribution")
    
    group.add_argument('--observ',type=str, choices = {'standard', "threshold"})

    # Creation of a mutually exclusive group for --id and --all
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--id",
        type=int,
        help="Image ID of the dataset of test of known_pattern . Required in --phase 'miss_classification",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Show paths of miss_classification images. Required in --phase 'miss_classification'",
    )

    args = parser.parse_args()

    #----------PARSE ARGS --------------------------------
    experiment_num = args.experiment
    file_config = args.config_file
    experiment = f"experiment_[{str(experiment_num)}]"
    distribution = args.distribution
    
    phase = args.phase
    observ = args.observ
    if phase =="visualization" or phase == "miss_classification":
        if observ is None:
            print(f"Phase {phase},  insert option observ standard | threshold")
            sys.exit()
        distribution = False
    ##############################
    #-----------------------------------------------------

    config = configparser.ConfigParser()
    config.read('config.ini')

    path_home_dataset = config["absolute_path"]["datasets"]
    print(f"Absolute path for datasets {path_home_dataset}")
 
    #-----------------------------------------------------


    #print(f"One of these must be active--------Observ {observ},  --- distribution  {distribution}")
    #-----------------------------------------------------
    ref_experiment = f"experiment_[{str(experiment_num)}]" 
    folder_experiment = f"experiment_[{str(experiment_num)}]"# experiment_[1]
    #-----------------------------------------------------
    #--------- CREATE or check FOLDER-------------------
    path_folder = os.path.join(root, folder_result)     # "./Results_open_set_testing"
    create_Folder(path_folder)  

    #--------- CHECK if FOLDER EXPERIMENT EXISTS-------------------
    path_folder_exp = os.path.join(path_folder, folder_experiment ) #./Results_open_set_testing/experiment_[1]
    create_Folder(path_folder_exp)

    #--------- LOAD FILE of CONFIGURATION ----------------------
    path_file_config = os.path.join(path_folder_exp,file_config) # path_file_config ->  ./Results_open_set_testing/ref_experiment_[1]/config.json
    print(Fore.CYAN+f"{path_file_config}"+Fore.RESET)
    try:
        with open(path_file_config, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(Fore.RED+f"The configuration file [{file_config}] is not present in the folder [{folder_result}/{folder_experiment}]"+Fore.RESET )
        print(Fore.MAGENTA+f"Creating the configuration file [{file_config}]... "+Fore.RESET)
        header = {"project": "known_unknown_classes","description": "Configuration file for open-set testing"}
        create_file_json_open_set_testing(path_file_config)
        print(Fore.MAGENTA+f"[X] Please compile the file [{file_config}]"+Fore.RESET)
        sys.exit()
    except Exception as e:
        print(Fore.RED+f"Error {e}"+Fore.RESET)
        sys.exit()
   
    #------------------- MISSCLASSIFIED ---------------------------------------------------
    # Conditional check: if phase is "missclassified", --id is required
    #print(f"id {args.id} -all {args.all}")
    if args.phase == "miss_classification" and ( args.id is None and args.all is False):
        print("Insert option --id <num> or --all")
        sys.exit(1)

    miss_class_id_or_all = None

    if args.phase == "miss_classification":
        if args.id is not None:
            miss_class_id_or_all = args.id
        elif args.all is not None:
            miss_class_id_or_all = args.all

        print(f"[OPTION] Missclassification {miss_class_id_or_all}") 
    
    if  distribution == True:
        distribution_test_bool = True
    else:
        distribution_test_bool = False


    #--------------------------------------------------------------------------------------
    #--------- READ FILE OF CONFIGURATION -----------------

    known_photos_folder = config.get("known_photos_folder") # "Imagenet_photos"
    known_classes_name = config.get("known_classes_name") # ["bookcase","gorilla", "tiger","umbrella"]
    known_assign_place = config.get("known_assign_place")
    dataset_known = config.get("src_dataset_known") # IMAGENET 
    dataset_unknown = config.get("src_dataset_unknown") # IMAGENET
    #dataset = config.get("src_dataset") # IMAGENET

    unknown_photos_folder = config.get("unknown_photos_folder") # "Imagenet_unknown" "Mnist_unknown"
    unknown_classes_name = config.get("unknown_classes_name") # ["hamset","lemon", "sock"]
    
    reciprocal_pattern_folder = config.get("reciprocal_pattern_folder") # "Mnist_pattern"
    reciprocal_classes_name = config.get("reciprocal_classes_name") # ["reciprocal_, reciprocal_1, reciprocal_2]
    reciprocal_assign_place = config.get("reciprocal_assign_place")

    if reciprocal_pattern_folder is not None:
        folder_pattern = os.path.join(path_home_dataset,reciprocal_pattern_folder )# "./Imagenet_patterns"
        pattern_experiment = config.get("pattern_experiment")
        if pattern_experiment is None:
            path_pattern = os.path.join(folder_pattern, f"pattern_{experiment}")  #"./Imagenet_patterns/pattern_experiment_[1]"
        else: 
            path_pattern = os.path.join(folder_pattern, pattern_experiment)

        print(f"Folder pattern {path_pattern}")
        if check_exists(path_pattern):
            print(Fore.GREEN+F"[{path_pattern}] EXIST"+Fore.RESET)
        else:
            print(Fore.RED+f"Folder PATTERN is not exist'"+Fore.RESET)
            sys.exit()
    else:
        print(Fore.RED+f"Insert information of Folder PATTERN'"+Fore.RESET)
        sys.exit()

    if known_assign_place is None:
        print(Fore.RED+f"Insert information of 'known_assign_place'"+Fore.RESET)
        sys.exit()
    if reciprocal_assign_place is None:
        print(Fore.RED+f"Insert information of 'reciprocal_assign_place'"+Fore.RESET)
        sys.exit()




   #------------------------------------------------------------------------------------------
    #-------- READ key [ref_experiment] in CONFIGURATION file ------------
    dropout= None
    radius_diverse  = False
    if config.get("ref_experiment"):
        exp_number = config["ref_experiment"].get("exp_num") # "experiment_[1]"
        print(f"EXPERIMENT {exp_number}")
        #if ref_experiment == exp_number: #                      ref_experiment ->  "experiment_[1]" == "experiment_[1]"
        name_architecture = config["ref_experiment"].get("architecture_obj") # "RenNet18_IMAGENET"
        network = config["ref_experiment"].get("model_type") # "ResNet18"
        model_filename = config["ref_experiment"].get("filename_model", f"model_{ref_experiment}.pth")
        dropout = config["ref_experiment"].get("dropout", None)
        root_results_folder = config["ref_experiment"].get("root_results_folder", f"Results_dataset_known_pattern")
        path_folder_ref = os.path.join(root, root_results_folder)
        path_mod = os.path.join(path_folder_ref, exp_number)

        path_model = os.path.join(path_mod, model_filename)
        print(f"Path model {path_model}")
        
        pretrained = config["ref_experiment"].get("pretrained")
        balanced = config["ref_experiment"].get("balanced")
        radius_diverse = config["ref_experiment"].get("radius_diverse", False)
        print(f"Radius diverse {radius_diverse}")

        if balanced is None:
            print(f"Insert information in key  'ref_experiment.balanced'")
            sys.exit()

        if name_architecture is None:
            print(f"Insert information in key  'ref_experiment.architecture_obj'")
            sys.exit()
        if network is None:
            print(f"Insert information in key  'ref_experiment.model_type'")
            sys.exit()

        if dataset_known is None or dataset_unknown is None:
            print(f"Insert information in key 'src_dataset_known {dataset_known}, or 'src_dataset_unknown {dataset_unknown} ")
            sys.exit()
    else:
        print(f"Insert information about 'ref_experiment")
        sys.exit()
        
    if  config.get("task_classification"):
        info_task_classification = config.get("task_classification") 
        print(info_task_classification)
        if info_task_classification.get("num_known_classes"):
            num_known_classes = info_task_classification.get("num_known_classes") 
        else:
            print(f"Insert information in key 'task_classification.num_known_classes")
            sys.exit()

        if info_task_classification.get("num_unknown_classes"):
            num_unknown_classes = info_task_classification.get("num_unknown_classes") 
        else:
            print(f"Insert information in key 'task_classification.num_unknown_classes")
            sys.exit()
            
        if info_task_classification.get("num_reciprocal_classes"):
            num_reciprocal_classes = info_task_classification.get("num_reciprocal_classes") 
        else:
            print(f"Insert information in key 'task_classification.num_reciprocal_classes")
            sys.exit()

        num_tot_classes = num_known_classes + num_unknown_classes + num_reciprocal_classes
        num_classes_K_R = num_known_classes + num_reciprocal_classes
    else:
            print(f"Insert information in key 'task_classiflication")
            sys.exit()

    
    print(Fore.CYAN+f"Informazion about network {name_architecture}, {network}, {dataset_known}, K: {num_known_classes}, R: {num_reciprocal_classes}, UK: {num_unknown_classes}"+Fore.RESET)
    #--------------------------------------------------------------------------

    #-------------------------------------------------
    if phase == 'miss_classification':
        path_file_miss = os.path.join(path_folder_exp, f"miss_classified_{exp_number}_{observ}.json")
        show_missclassification(path_file_miss, exp_number, miss_class_id_or_all)
        sys.exit()

    #--------------------------------FOLDER of PHOTOS [KNOWN]----------------------------------------------
    if known_photos_folder is None :
        print(Fore.RED+f"Insert information about 'known_photos_folder'"+Fore.RESET)
        sys.exit()
    elif known_classes_name is None:
        print(Fore.RED+f"Insert information in key 'known_classes_name'"+Fore.RESET)
        sys.exit()
    else:
        known_path_photos = os.path.join(path_home_dataset,known_photos_folder) #"./Imagenet_photos"
    #---------------------------------FOLDER of PHOTOS [UNKNOWN]----------------------------------------
    if unknown_photos_folder is None:
        print(Fore.RED+f"Insert information about 'unknown_photos_folder'"+Fore.RESET)
        sys.exit()
    elif unknown_classes_name is None:
        print(Fore.RED+f"Insert information in key 'unknown_classes_name'"+Fore.RESET)
        sys.exit()
    else:
        unknown_path_photos = os.path.join(path_home_dataset,unknown_photos_folder)

    #---------------------------------FOLDER of RECIPROCAL [PETTERN]----------------------------------------
    if reciprocal_pattern_folder is None:
        print(Fore.RED+f"Insert information in key 'reciprocal_pattern_folder'"+Fore.RESET)
        sys.exit()
    elif reciprocal_classes_name is None:
        print(Fore.RED+f"Insert information in key 'reciprocal_pattern_folder'"+Fore.RESET)
        sys.exit()
    else:
        rpp = os.path.join(path_home_dataset,reciprocal_pattern_folder)
        reciprocal_path_patterns = path_pattern
    
    

    print(f"INFORMATION CLASSES {known_path_photos} {known_classes_name} \n {unknown_photos_folder}, {unknown_classes_name} \n {reciprocal_pattern_folder} , {reciprocal_classes_name}")
    print(f"INFORMATION {name_architecture}, {network}, {model_filename} ,{dataset_unknown},{dataset_known},{path_model}, TOT = K+UK+R:{num_tot_classes} ")

#################################### transformer -------
    transformer_obj = Transformer(seed, generator)


    transforms_all = transformer_obj.get_transforms( dataset_known, network, bool_all= True, bool_train=None) # traformazione per tutto il dataset
    transforms_train = transformer_obj.get_transforms( dataset_known, network, bool_all= None, bool_train=True) # trasformazione per il dataset di train
    transforms_test = transformer_obj.get_transforms( dataset_known, network, bool_all= None, bool_train=False) # trasformazione per il dataset di test


#------------------------------------------ INITIALIZE DATASET CREATOR ----------------------------
#----------- initialize Dataset_Creator------------
    dataset_creator_obj = Dataset_Creator(seed, generator)

    ###------- [KNOWN] --------------------------------------
    dataset_creator_obj.setFolder_Photos_Known(known_path_photos)
    for n in known_classes_name:
        print(f"known class name {n}")
        dataset_creator_obj.add_known_class_name(n)

    ###-------[RECIPROCAL] ----------------------------------
    dataset_creator_obj.setFolder_Pattern(reciprocal_path_patterns)
    for n in reciprocal_classes_name:
        print(f"Recipocal_class name {n}")
        dataset_creator_obj.add_reciprocal_class_name(n)

    ##-------[  UNKNOWN ] -------------------------------------
    dataset_creator_obj.setFolder_Photos_Unknown(unknown_path_photos)
    for n in unknown_classes_name:
        print(f"UNknown class name {n}")
        dataset_creator_obj.add_unknown_class_name(n)
    
    if known_assign_place is not None:
        dataset_creator_obj.set_known_assign_place(known_assign_place) 
    if reciprocal_assign_place is not None:
        dataset_creator_obj.set_reciprocal_assign_place(reciprocal_assign_place) 
    
    ############------------------------------------##########
    ############       Create Dataset KNOWN         ##########
    ############------------------------------------##########
    if dataset_known == "IMAGENET":
        dataset_creator_obj.create_dataset_from_ImageFolder(type_task = "known", validation_size=0.15, test_size=0.15, transforms_all = transforms_all, transforms_train= transforms_train, transforms_test=transforms_test) # effettua anche lo splits

        
    elif dataset_known == "MNIST":
        create_Folder(known_path_photos)

        path_dataset_download = os.path.join(path_home_dataset, "MNIST_data")
        dataset_creator_obj.create_dataset_from_Datasets(root=path_dataset_download, type_task ="known", type_dataset = dataset_known, validation_size=0.15, transforms_all = transforms_all, transforms_train = transforms_train, transforms_test = transforms_test)
    

    for type_t in ["train_validation","train", "validation", "test", "overall"]:
        type_dataset = f"known_{type_t}"
        print(type_dataset)
        count_sample = dataset_creator_obj.count_sample_in_classes(type_dataset = type_dataset)
        if count_sample is not None:
            add_update_key(path_file_config, "known",f"counter_{type_t}", count_sample)

    
    map_class_to_label = dataset_creator_obj.get_known_map_name_to_label()
    map_label_to_class = dataset_creator_obj.get_known_map_label_to_name()

    add_update_key(path_file_config, "known" ,"className_to_label", map_class_to_label)
    add_update_key(path_file_config, "known" ,"label_to_className", map_label_to_class)
    
    #--------- SHOW SAMPLE INFO  --------------------------------------------------
    
    for idx in [100, 300, 100]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("train_known", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[train_known] - {path_img} - {label} - {class_name}"+Fore.RESET)
    for idx in [10, 200, 500]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("validation_known", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[validation_known] - {path_img} - {label} - {class_name}"+Fore.RESET)
    for idx in [40, 700, 110]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("test_known", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[test_known] - {path_img} - {label} - {class_name}"+Fore.RESET)
    
    #########################################################################
    file_csv_path_img = f"for_open_set_{dataset_known}_known_filename_img.csv"
    df_train = dataset_creator_obj.get_dict_path_img("train_known")
    df_validation = dataset_creator_obj.get_dict_path_img("validation_known")
    df_test =dataset_creator_obj.get_dict_path_img("test_known")

    path_csv_path_img = os.path.join(path_home_dataset, file_csv_path_img)
    df_union = pd.concat([df_train, df_validation, df_test], ignore_index= True)
    df_union.to_csv(path_csv_path_img, index=False)

    #######################################################################
    #--------------- DDefine the number of samples for each class and then those to be extracted from the test set ########à
    # Get the number of samples of a KNOWN class (from the train + validation + test dataset)
    count_sample_for_class = dataset_creator_obj.count_sample_in_classes(type_dataset = "known_overall")
    print(f"Numero di sample per classe in tutto il set known {count_sample_for_class}")
    num_samples_in_class_known = int(count_sample_for_class.get(next(iter(count_sample_for_class))))
    print(f"Numero di elementi in una classe nota {num_samples_in_class_known }")
    if balanced == False and isinstance(count_sample_for_class, dict):
        print("Le classi NOTE non sono bilanciate: determina la somma ")
        total_known  = sum(count_sample_for_class.values())

    # Define the number of samples to consider in the TEST sets of unknown and reciprocal classes as: (num_samples_in_known_class * num_reciprocal_classes) / (num_reciprocal_classes + num_unknown_classes)
    count_sample_for_class_test = dataset_creator_obj.count_sample_in_classes(type_dataset = "known_test")
    print(f"Number of samples per class in the TEST set (known) {count_sample_for_class_test}")
    if balanced == True:
        num_samples_for_class_in_test_known = int(count_sample_for_class_test.get(next(iter(count_sample_for_class_test))))
        print(f"Number of elements in a known class of the TEST set. {num_samples_for_class_in_test_known }")
        max_for_class_in_test_set_recip_unknown = int((num_samples_for_class_in_test_known * num_reciprocal_classes)/(num_reciprocal_classes + num_unknown_classes))
        print(f"({num_samples_for_class_in_test_known} * {num_reciprocal_classes})/({num_reciprocal_classes + num_unknown_classes})= {max_for_class_in_test_set_recip_unknown}")
    else:
        total_known_test = sum(count_sample_for_class_test.values())
        print(f"Total number of elements in the KNOWN classes of the TEST set. {total_known_test }")
        max_for_class_in_test_set_recip_unknown = int((total_known_test)/(num_reciprocal_classes + num_unknown_classes))
        print(f"({total_known_test})/({num_reciprocal_classes + num_unknown_classes}) = {max_for_class_in_test_set_recip_unknown}")


    print(f"Number of samples to extract from each class from the unknown and test patterns in the TEST set.: {max_for_class_in_test_set_recip_unknown}")
    
    ############------------------------------------------------#############
    ########        Create Dataset Reciprocal-class PATTERN         #########
    ############------------------------------------------------#############
    ###################################################################################################################
    print("Create dataset reciprocal from test for set unknown")
    if reciprocal_assign_place is None: 
        label_mapping_reciprocal_class = {}  # eventuamente  si potrebbe fare la lettura da file # se non è presente accoda dalla fine 
    
        if len(label_mapping_reciprocal_class.keys()) == 0:
            next_label = num_known_classes
            list_label_pattern = list(range(next_label, next_label + num_reciprocal_classes))
            label_mapping_reciprocal_class = { idx:label for idx, label in zip(range(num_reciprocal_classes),list_label_pattern) }
            print(Fore.MAGENTA+f"{label_mapping_reciprocal_class}"+Fore.RESET)
    else:
        label_mapping_reciprocal_class = { int(str_idx):place  for str_idx, place in reciprocal_assign_place.items()  }
        print(Fore.MAGENTA+f"Mapp for reciprocals : {label_mapping_reciprocal_class}"+Fore.RESET)

    create_Folder(known_path_photos)
    if dataset_known == "MNIST":
        validation_size = 0.13 
        test_size = 0.14
    else:
        validation_size = 0.15
        test_size = 0.15

    num_samples_for_classes = None
    print(f"Balance the number of patterns ? {balanced}")
    if balanced == False:
        print("Unbalanced patterns")
        num_samples_for_classes = count_sample_for_class #Use this in case class balancing is not desired; thus, the number of patterns to take for a class is equal to the samples of the corresponding known class
    
    extract_sample_for_class_test = max_for_class_in_test_set_recip_unknown
    dataset_creator_obj.create_dataset_from_ImageFolder(type_task = "reciprocal_class", validation_size=validation_size, test_size=test_size, bool_balance = balanced ,transforms_all = transforms_all, transforms_train= transforms_train, transforms_test=transforms_test, label_mapping=label_mapping_reciprocal_class, num_samples_in_class= num_samples_in_class_known, extract_sample_for_class_test = extract_sample_for_class_test,  num_samples_for_classes=num_samples_for_classes) # effettua anche lo splits

    for type_t in ["train", "validation","train_validation", "test","overall"]:
            count_sample = dataset_creator_obj.count_sample_in_classes(type_dataset = f"reciprocal_{type_t}")
            if count_sample is not None:
                add_update_key(path_file_config, "reciprocal_class",f"counter_{type_t}", count_sample)

    map_class_to_label_reciprocal = dataset_creator_obj.get_reciprocal_map_name_to_label()
    map_label_to_class_reciprocal = dataset_creator_obj.get_reciprocal_map_label_to_name()

    add_update_key(path_file_config, "reciprocal_class" ,"className_to_label", map_class_to_label_reciprocal)
    add_update_key(path_file_config, "reciprocal_class" ,"label_to_className", map_label_to_class_reciprocal)

    #--------- SHOW SAMPLE INFO  --------------------------------------------------
    
    for idx in [100, 300, 100]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("train_reciprocal", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[train_reciprocal] - {path_img} - {label} - {class_name}"+Fore.RESET)
    for idx in [10, 200, 500]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("validation_reciprocal", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[validation_reciprocal] - {path_img} - {label} - {class_name}"+Fore.RESET)
    for idx in [40, 700, 110]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("test_reciprocal", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[test_reciprocal] - {path_img} - {label} - {class_name}"+Fore.RESET)

    
    ############------------------------------------------------#############
    ########        Create Dataset Unknown-class                   ##########
    ############------------------------------------------------#############
    print("Create dataset unknown")
    
    type_t = "test"
    create_Folder(unknown_path_photos)
    path_dataset_download = os.path.join(path_home_dataset, f"{dataset_unknown}_data")
    if dataset_unknown == "IMAGENET":
        dataset_creator_obj.create_dataset_from_ImageFolder(type_task = "unknown", validation_size=0.15, test_size=0.15, transforms_all = transforms_all, transforms_train= transforms_train, transforms_test=transforms_test,extract_sample_for_class_test = extract_sample_for_class_test) # effettua anche lo splits
       
    elif dataset_unknown == "MNIST":
        dataset_creator_obj.create_dataset_from_Datasets(root=path_dataset_download, type_task ="unknown", type_dataset = dataset_unknown, validation_size=0.15, transforms_all = transforms_all, transforms_train = transforms_train, transforms_test = transforms_test, max_num_sample= None, extract_sample_for_class_test = extract_sample_for_class_test)
    
    count_sample = dataset_creator_obj.count_sample_in_classes(type_dataset = f"unknown_{type_t}")
    print(f"Counter unknown {count_sample}")
    if count_sample is not None:
        add_update_key(path_file_config, "unknown_class",f"counter_{type_t}", count_sample)
        
    map_class_to_label_unknown = dataset_creator_obj.get_unknown_map_name_to_label()
    add_update_key(path_file_config, "unknown_class" ,"className_to_label", map_class_to_label_unknown)

    #--------- SHOW SAMPLE INFO  --------------------------------------------------
    for idx in [50, 300, 1500]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("test_unknown", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[test_unknown] - {path_img} - {label} - {class_name}"+Fore.RESET)

    #-----------------------------------------------------------------------------------------------
    #--------------------- concatenate test datasets -------------------------------------------
    dataset_creator_obj.concat_sets("known_reciprocal_unknown")

    map_class_to_label_known_reciprocal_unknown = dataset_creator_obj.get_known_reciprocal_unknown_map_name_to_label()

    add_update_key(path_file_config, "known_reciprocal_unknown" ,"className_to_label", map_class_to_label_known_reciprocal_unknown)
    ##############
    if config.get("legend_for_plot") is None:
        print("Legend_for_plot is not present in the configuration file")
        map_label_to_class_known_reciprocal = dataset_creator_obj.get_known_reciprocal_map_label_to_name() 
        map_legend_for_matrix = map_label_to_class_known_reciprocal | {8 :"unknown"}
    else:
        print("Legend_for_plot is present in the configuration file")
        map_dict = config.get("legend_for_plot")
        
        map_legend_for_matrix = {int(key):value for key,value in map_dict.items()}

    print(f"Map LEGEND MATRIX  {map_legend_for_matrix}")
    #############


    #--------- SHOW SAMPLE INFO  del dataset CONCATENATO--------------------------------------------------

    #for idx in [800, 20,40,1000,300,75,98,47,685, 5800, 200, 400]:
    for idx in [80, 20,40]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("test_known_reciprocal_unknown", idx)
        if path_img is not None:
            print(Fore.RED+f"[test_known_reciprocal_unknown] - {path_img} - {label} - {class_name}"+Fore.RESET)

    #---------------------------------------------------------------------------------
    num_classes_K_R = num_known_classes + num_reciprocal_classes
    ##----------- NETWORK  --------------------------------------------
    manager_network = Manager_Networks()
    if pretrained == True:
        print(f"Pretrained TRUE- class num tot {num_classes_K_R}")
        net_obj = manager_network.get_network_pretrained(network, dataset_known, num_classes_K_R,dropout)
        model = net_obj.get_model()

    elif pretrained == False:
        print(f"Pretrained FALSE - class num tot {num_known_classes + num_reciprocal_classes}")
        
        net_obj = manager_network.get_network_no_pretrained(network, dataset_known, num_classes_K_R,dropout)
        print("Nome criterion class",net_obj.get_criterion_name())
        model = net_obj
    

    classes_index_reciprocal = config.get("idx_reciprocal_class")
    if classes_index_reciprocal is None:
        print("Insert information about idx_reciprocal_class")
        sys.exit()
        
    print(f"classes_index_reciprocal {classes_index_reciprocal}")
    replace_idx_unknown = num_classes_K_R

   
    #list_max = None
    list_min = None 
    list_distance = None
    #print(path_mod)
    
    if observ == "threshold": # READING INSIDE THE CONFIGURATION FILE
        # lreading the IDEAL distribution of the unknown file with patterns 
        file_distribution_ref_exp = os.path.join(path_mod, "config.json" )
        print(f"[app_known_pattern] PERCORSO DI RIFERIEMNTO DISTRIBUZIONE {file_distribution_ref_exp}")
        print(exp_number)
        #read_distribution_1 = read_distribution(file_distribution_ref_exp, exp_number, "distribution_report_set_test")
        read_distribution_1 = read_distribution(file_distribution_ref_exp, exp_number, "distribution_report_set_validation")
        #read_distribution_2 = read_distribution(file_distribution_ref_exp, exp_number, "distribution_report_set_training")
        list_min_1 = list_min_prob_for_classes(read_distribution_1)
        #list_min_2 = list_min_prob_for_classes(read_distribution_2)
        #list_min  = [(a + b) / 2 for a, b in zip(list_min_1, list_min_2)]
        list_min = list_min_1
        

    ##############################################################################################

    if phase == "testing": 
        
        #---------------------------- create dataset Loader PER DATASET CONCATENATE ------------------------------   
        info_experiment = config["ref_experiment"]
        
        batch_size = info_experiment.get("batch_size", 32)
        
        

        add_update_key(path_file_config, experiment, "batch_size", batch_size )
        

        #-----Create and Get DATALOADER -----------------------------
        dataset_creator_obj.create_DataLoader("test_known_reciprocal_unknown",batch_size)
        
        #-------------- test determinisctoco  
        print(f"TEST DETERMINISTICO SU  test open set LOADER")
        load_test = dataset_creator_obj.get_datasetLoader("test_known_reciprocal_unknown")
        for img, label in load_test:
            print(f"{label}")
            break

        load_test = dataset_creator_obj.get_datasetLoader("test_known_reciprocal_unknown")
        for img, label in load_test:
            print(f"{label}")
            break


        

       
        path_distribution = None

        if distribution_test_bool:
            path_distribution = os.path.join(path_folder_exp, f"distribution_prob_test_{exp_number}_[origin_pop].csv")


        
        if observ is None:
            observ = "standard"
       
        run_testing( model, dataset_creator_obj, batch_size,num_classes_K_R, map_legend_for_matrix, phase = phase, type_dataset = "test_known_reciprocal_unknown", experiment = experiment, path_model= path_model , path_dest = path_folder_exp, path_file_config = path_file_config, distribution_test_bool = distribution_test_bool, path_distribution = path_distribution, read_distribution=list_min, read_distance = list_distance, distr_max_unknown=None, observ = observ, classes_index_reciprocal=classes_index_reciprocal, replace_idx_unknown=replace_idx_unknown )


    if phase == 'visualization':
        index_unknown = num_classes_K_R
        path_model = os.path.join(path_mod, model_filename)
        print(f"Path model {path_model}")
        info_experiment = config["ref_experiment"]
        batch_size = info_experiment.get("batch_size", 32)
        legend_for_matrix = config.get("legend_for_plot")
        if legend_for_matrix is None:
            legend_for_matrix = map_label_to_class_known_reciprocal | {index_unknown:"unknown"}
        name_architecture_obj = net_obj.__class__.__name__
        print(f"Name architecture: {name_architecture_obj}")
        print(f"Legend : {legend_for_matrix}")
        #print(model)

        legend_for_visualizing = { int(str_ind): name for str_ind, name  in legend_for_matrix.items()}
        print(f"Legend for visualizing int: {legend_for_visualizing}")

        idx_reciprocal_class = config.get("idx_reciprocal_class")
        if idx_reciprocal_class is None:
            print(Fore.RED+f"Insert in file configuration idx_reciprocal_class"+Fore.RESET)
            sys.exit()
        suffix = f"Open_set_test_[{experiment_num}]_[{dataset_known}]_[{observ}]"
        
        
        idx_known_class = [int(key) for key in known_assign_place.keys()]
        print(f"idx_known_class {idx_known_class} - indx_unknown {index_unknown}")
        print(Fore.CYAN+f"--------- Processing VISUALIZING -----------------------"+Fore.RESET)
        run_visualizing( model, name_architecture_obj, dataset_creator_obj, batch_size, phase = phase, type_dataset = "test_known_reciprocal_unknown", num_classes= num_classes_K_R, experiment = experiment, path_model= path_model , path_dest = path_folder_exp, path_file_config = path_file_config, map_label_to_class = legend_for_matrix, idx_reciprocal_class=idx_reciprocal_class,idx_known_class =idx_known_class, idx_unknown = index_unknown, outlier=True , suffix=suffix, list_min=list_min, classes_index_reciprocal=classes_index_reciprocal, replace_idx_unknown=replace_idx_unknown, observ =observ,list_distance = list_distance, bool_openset_testing=True )
        legend_for_demo = dataset_creator_obj.known_map_label_to_name  | {index_unknown:"unknown"}
        print(Fore.GREEN+f"--------- Processing VISUALIZING DEMO-----------------------"+Fore.RESET)
        run_visualizing( model, name_architecture_obj, dataset_creator_obj, batch_size, phase = phase, type_dataset = "test_known_reciprocal_unknown", num_classes= num_classes_K_R, experiment = experiment, path_model= path_model , path_dest = path_folder_exp, path_file_config = path_file_config, map_label_to_class = legend_for_demo, idx_reciprocal_class=idx_reciprocal_class,idx_known_class =idx_known_class, idx_unknown = index_unknown, outlier=True , suffix=suffix, list_min=list_min, classes_index_reciprocal=classes_index_reciprocal, replace_idx_unknown=replace_idx_unknown, observ =observ,list_distance = list_distance, bool_openset_testing=True, demo=True )


    if distribution_test_bool == True:

        # read file csv and calculate the threshold min and max of probs

        try:
            df_distribution = pd.read_csv(path_distribution)
            distribution_report = { }
            for class_id in range(num_classes_K_R):
                # Extract only the rows with ID_CLASS = class_id.
                df_class = df_distribution[ df_distribution["ID_CLASS"] == class_id ]
                # Calculate the minimum and maximum for each column related to the probabilities
                dict_prob = {}
                for prob_id in range(num_classes_K_R):
                    min_prob = df_class[f"prob_{prob_id}"].min() 
                    max_prob = df_class[f"prob_{prob_id}"].max()
                    dict_prob[f"{prob_id}"] = {"min": min_prob, "max": max_prob}

                distribution_report[f"class_{class_id}"] = dict_prob 

            # save distribution report in file config
            add_update_key(path_file_config, experiment, f"distribution_report_set_{phase}", distribution_report )
            

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
