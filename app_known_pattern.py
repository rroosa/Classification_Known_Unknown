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
from colorama import Fore
import configparser

root = os.getcwd()
folder_result = "Results_dataset_known_pattern"
folder_result_known =  "Results_dataset_known"
logdir_root = "logs_fitness_know_reciprocal"
port_logs = 6008
path_home_dataset = None
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

num_classes_tot = None
num_classes_known = None
num_classes_pattern = None

def set_seed(seed):
    print("set SEED")
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
    parser = argparse.ArgumentParser(description='Classification known classes')
    parser.add_argument('--experiment', type=int, choices={1,2}, required= True)
    parser.add_argument('--config_file', type=str, default= "config.json")
    parser.add_argument('--distribution', action="store_true",help="Collect distribution features for classes"  )
    parser.add_argument('--phase', type=str, choices = {'training', 'test', "visualization", "miss_classification"}, required= True)
    parser.add_argument('--continue_training',  action="store_true")
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
    distribution_test_bool = args.distribution
    phase = args.phase
    continue_training = args.continue_training
    #-----------------------------------------------------
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_home_dataset = config["absolute_path"]["datasets"]
    print(f"Absolute path for datasets {path_home_dataset}")
 
    #-----------------------------------------------------

    if args.phase == "miss_classification" and ( args.id is None and args.all is False):
        print("Insert option --id <num> or --all")
        sys.exit(1)

    miss_class_id_or_all = None

    if args.phase == "miss_classification":
        if args.id is not None:
            miss_class_id_or_all = args.id
        elif args.all is not None:
            miss_class_id_or_all = args.all


    num_classes = None
    num_classes_unknown = None
    
    #--------- CREATE or check FOLDER-------------------
    path_folder = os.path.join(root, folder_result) # "Results_dataset_known_pattern"
    create_Folder(path_folder)  

    #--------- CHECK EXISTS FOLDER EXPERIMENT -------------------
    path_folder_exp = os.path.join(path_folder, experiment ) #"Results_dataset_known_pattern/ experiment[1]"
    create_Folder(path_folder_exp)

    #--------- LOAD FILE of CONFIGURATION ----------------------
    path_file_config = os.path.join(path_folder_exp,file_config)
    print(Fore.BLUE+f"Path file config {path_file_config}"+Fore.RESET)

    try:
        with open(path_file_config, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"File config.json is not present, insert it in folder [{folder_result}/{experiment}]" )
        create_file_json_known_pattern(path_file_config)
        sys.exit()
    except Exception as e:
        print(f"Error {e}")
        sys.exit()
    

    #--------- READ FILE OF CONFIGURATION -----------------
    folder_photos = config.get("folder_photos") # "Imagenet_photos"
    known_name_classes = config.get("known_name_classes") # ["bookcase","gorilla", "tiger","umbrella"]
    known_assign_place = config.get("known_assign_place") # {"six":0, "zero":2, }
    path = os.path.join(path_home_dataset,folder_photos) #"./Imagenet_photos"
    dataset = config.get("src_dataset") # IMAGENET
    idx_reciprocal_class = config.get("idx_reciprocal_class")
    folder_pattern_root = config.get("folder_pattern_root") # "Imagenet_patterns"
    if folder_pattern_root is not None:
        folder_pattern = os.path.join(path_home_dataset,folder_pattern_root )# "./Imagenet_patterns"
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
    
    if idx_reciprocal_class is None:
        print(Fore.RED+f"Insert information of idx_reciprocal_class'"+Fore.RESET)
        sys.exit()

    reciprocal_class_pattern = config.get("reciprocal_name_pattern") # ["reciprocal_all"]
    reciprocal_assign_place = config.get("reciprocal_assign_place")
    print(f"known_assign_place {known_assign_place}")

    info_experiment = config.get(experiment)
    pretrained = None
    transfer_learning_feature_based = None
    transfer_model_num = None
    transfer_num_class= None
    path_model_transfer = None
    if info_experiment is not None:
        balanced = info_experiment.get("balanced")
        network = info_experiment.get("network").get("architecture") 
        pretrained = info_experiment.get("network").get("pretrained")
        pretrained = pretrained if isinstance(pretrained, bool) else bool(strtobool(pretrained.strip().lower()))   if isinstance(pretrained, str) else False 
        dropout = info_experiment.get("network").get("dropout", None)
        print(f"Pretrained: {type(pretrained)}, {pretrained}, dropout: {dropout}")
        transfer_learning_feature_based = info_experiment.get("network").get("transfer_learning_feature_based")
        if network is  None or pretrained is  None:
            print(f"Insert information under key 'experiment_[{str(experiment_num)}]':'network.achitecture , network.pretrained'")
            sys.exit()
        #------------------------- TRANSFER LEARNING FEATURES --------------------------------
        if transfer_learning_feature_based is not None:
            transfer_model_num = info_experiment.get("network").get("transfer_model_num")

            transfer_num_class = info_experiment.get("network").get("transfer_num_class")
            if transfer_num_class is None or transfer_model_num is None:
                print(f"Insert information in the key 'network.transfer_num_class' and 'network.transfer_model_num'")
                sys.exit()
            
            path_model_transfer = os.path.join(folder_result_known, transfer_model_num )
            path_model_transfer = os.path.join(path_model_transfer, f"model_{transfer_model_num}.pth" )
            print(f"Path_model_transfer {path_model_transfer}")
        if balanced is None:
            print(f"Insert information in the key experiment[{experiment_num}].balanced'")
            sys.exit()

        #------------------------- -----------------------------------------------------------

       
  
    else:
        print(f"Insert information the key 'experiment_[{str(experiment_num)}]':'network.achitecture , network.pretrained'")
        sys.exit()
    
    
    if folder_photos is None:
        print(f"Insert information about 'folder_photos'")
        sys.exit()
    if known_name_classes is None:
        print(f"Insert information about 'known_name_classes'")
        sys.exit()
    else:
        num_classes_known = len(known_name_classes)
        print(f"Num classes: {num_classes}")
    if dataset is None:
        print(f"Insert information about 'src_dataset'")
        sys.exit()
    
    if reciprocal_class_pattern is None:
        print(f"Insert information about 'reciprocal_name_pattern'")
        sys.exit()
    else: 
        num_reciprocal_classes = len(reciprocal_class_pattern)
 

    num_classes_tot = num_classes_known + num_reciprocal_classes
    print(f"Num classe known:{num_classes_known}, Num classe unknown: {num_reciprocal_classes}, Numclasses tot {num_classes_tot}")

    #-------------------------------------------------
    observ="standard"
    if phase == 'miss_classification':
        path_file_miss = os.path.join(path_folder_exp, f"miss_classified_{experiment}_{observ}.json")
        show_missclassification(path_file_miss, experiment, miss_class_id_or_all)
        sys.exit()
        
    #----------- initialize Dataset_Creator------------
  
    dataset_creator_obj = Dataset_Creator(seed, generator)
    dataset_creator_obj.setFolder_Photos_Known(path) 
    dataset_creator_obj.setFolder_Pattern(path_pattern) 
    if known_assign_place is not None:
        dataset_creator_obj.set_known_assign_place(known_assign_place) 
    if reciprocal_assign_place is not None:
        dataset_creator_obj.set_reciprocal_assign_place(reciprocal_assign_place) 


    for n in known_name_classes:
        print(f"known class name {n}")
        dataset_creator_obj.add_known_class_name(n)
    
    for n in reciprocal_class_pattern:
        print(f"Recipocal_class pattern name {n}")
        dataset_creator_obj.add_reciprocal_class_name(n)

    #---------- Tranformer-------------------------
    transformer_obj = Transformer(seed, generator)


    transforms_all = transformer_obj.get_transforms( dataset, network, bool_all= True, bool_train=None) # traforms for all il dataset
    transforms_train = transformer_obj.get_transforms( dataset, network, bool_all= None, bool_train=True) # trasforms for training set
    transforms_test = transformer_obj.get_transforms( dataset, network, bool_all= None, bool_train=False) # trasforms for test set
    print(transforms_train)

    ############------------------------------------##########
    ############       Create Dataset KNOWN         ##########
    ############------------------------------------##########
    if dataset == "IMAGENET":
        dataset_creator_obj.create_dataset_from_ImageFolder(type_task = "known", validation_size=0.15, test_size=0.15, transforms_all = transforms_all, transforms_train= transforms_train, transforms_test=transforms_test) # effettua anche lo splits
        logdir_root = "logs_known_pattern_IMAGENET"
        port = 6006

        
    elif dataset == "MNIST":
        Folder_Photos = f"{dataset}_photos"
        path_Folder_Photos = os.path.join(root, Folder_Photos)
        create_Folder(path_Folder_Photos)
        logdir_root = "logs_known_pattern_MNIST"
        port = 6008
        path_dataset_download = os.path.join(path_home_dataset, "MNIST_data")
        dataset_creator_obj.create_dataset_from_Datasets(root=path_dataset_download, type_task ="known", type_dataset = dataset, validation_size=0.15, transforms_all = transforms_all, transforms_train = transforms_train, transforms_test = transforms_test)
    


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
            print(Fore.MAGENTA+f"[validation_known] - {path_img} - {label} - {class_name}"+Fore.RESET)
    for idx in [40, 700, 110]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("test_known", idx)
        if path_img is not None:
            print(Fore.BLUE+f"[test_known] - {path_img} - {label} - {class_name}"+Fore.RESET)

    print("Qui")
    ############------------------------------------------------#############
    ########            Create Dataset Reciprocal-class PATTERN    ##########
    ############------------------------------------------------#############
    # Get the number of samples of a KNOWN class (from the train + validation + test dataset)
    count_sample_for_class = dataset_creator_obj.count_sample_in_classes(type_dataset = "known_overall")
    print(f"Numero di sample per classe in overall known {count_sample_for_class}")
    num_samples_in_class_known = count_sample.get(next(iter(count_sample)))
    print(f"Numero di elementi in una classe nota {num_samples_in_class_known }")

    if reciprocal_assign_place is None: 
        label_mapping_reciprocal_class = {}  #Optionally, reading from file could be done
                                            # if not present, append at the end
    
        if len(label_mapping_reciprocal_class.keys()) == 0:
            next_label = len(known_name_classes)
            list_label_pattern = list(range(next_label, next_label + num_reciprocal_classes))
            label_mapping_reciprocal_class = { idx:label for idx, label in zip(range(num_reciprocal_classes),list_label_pattern) }
            print(Fore.MAGENTA+f"{label_mapping_reciprocal_class}"+Fore.RESET)
    else:
        label_mapping_reciprocal_class = { int(str_idx):place  for str_idx, place in reciprocal_assign_place.items()  }
        print(Fore.MAGENTA+f"{label_mapping_reciprocal_class}"+Fore.RESET)

    
    if dataset == "MNIST":
        validation_size = 0.13 
        test_size = 0.14
    else:
        validation_size = 0.15
        test_size = 0.15

    num_samples_for_classes = None
    print(f"Bilancire il numero de patter ? {balanced}")
    if balanced == False:
        print("Pattern non bilanciati")
        num_samples_for_classes = count_sample_for_class  # Use this in case class balancing is not desired, so the number of patterns take

    dataset_creator_obj.create_dataset_from_ImageFolder(type_task = "reciprocal_class", validation_size=validation_size, test_size=test_size, bool_balance = balanced ,transforms_all = transforms_all, transforms_train= transforms_train, transforms_test=transforms_test, label_mapping=label_mapping_reciprocal_class, num_samples_in_class= num_samples_in_class_known, num_samples_for_classes=num_samples_for_classes) # effettua anche lo splits

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

    #--------------------- CONCATENARE i SET  train, validation, e test corrispondenti a KNOWN E reciprocal ------------------------
    dataset_creator_obj.concat_sets("known_reciprocal")

    map_class_to_label_known_reciprocal = dataset_creator_obj.get_known_reciprocal_map_name_to_label()
    map_label_to_class_known_reciprocal = dataset_creator_obj.get_known_reciprocal_map_label_to_name()

    add_update_key(path_file_config, "known_reciprocal" ,"className_to_label", map_class_to_label_known_reciprocal)
    add_update_key(path_file_config, "known_reciprocal" ,"label_to_className", map_label_to_class_known_reciprocal)



    #--------- SHOW SAMPLE INFO  del dataset CONCATENATO--------------------------------------------------
    
    for idx in [1800, 20000 , 500, 4000]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("train_known_reciprocal", idx)
        if path_img is not None:
            print(Fore.GREEN+f"[train_known_reciprocal] - {path_img} - {label} - {class_name}"+Fore.RESET)
    for idx in [50, 25, 65]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("validation_known_reciprocal", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[validation_known_reciprocal] - {path_img} - {label} - {class_name}"+Fore.RESET)
    for idx in [50,80, 24, 19]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("test_known_reciprocal", idx)
        if path_img is not None:
            print(Fore.RED+f"[test_known_reciprocal] - {path_img} - {label} - {class_name}"+Fore.RESET)

    #---------------------------------------------------------------------------------
    
    
    ##----------- NETWORK  --------------------------------------------
    manager_network = Manager_Networks()
    if transfer_learning_feature_based == True:
        num_class_considered = transfer_num_class 
    else:
        num_class_considered = num_classes_tot

    if pretrained == True:
        print(f"Pretrained TRUE - class num tot {num_class_considered}")
        net_obj = manager_network.get_network_pretrained(network, dataset, num_class_considered,dropout)
        print("Nome criterion class",net_obj.get_criterion_name())
        model = net_obj.get_model()
        
        

    elif pretrained == False:
        print(f"Pretrained FALSE - class num tot {num_class_considered}")
        net_obj = manager_network.get_network_no_pretrained(network, dataset, num_class_considered,dropout)
        print("Nome criterion class",net_obj.get_criterion_name())
        model = net_obj
   
    if transfer_learning_feature_based == True:
        net_obj.transfer_learning_feature_based(model, path_model_transfer,num_classes_tot, transfer_num_class )


    
    criterion = info_experiment.get("criterion")
    if criterion is not None:
        criterion_type = criterion.get("type")
        print(f"Criterion function {criterion_type}")
        if criterion_type is not None:
            if criterion_type == "JSD_Loss":
                net_obj.set_criterion(JSD_Loss())
            elif criterion_type == "CrossEntropyLoss":
                net_obj.set_criterion( nn.CrossEntropyLoss(reduction='mean'))

          


        else:
            print("Add key criterion.type")
            sys.exit()
                
    scheduler = info_experiment.get("scheduler")
    if scheduler is not None: 
        scheduler_type = scheduler.get("type")
        print(f"Scheduler function {scheduler_type}")
        if scheduler_type:
            net_obj.set_scheduler_name(scheduler_type)


    
    #------------ PHASE TRAINING ------------------------------------------------------
    if phase == 'training':

        #----- get info about hyperparameters e settings from file of configuration ----------
        hyperparameters = info_experiment.get("hyperparameters")
        if hyperparameters is not None:
            lr = hyperparameters.get("lr")
            weight_decay =  hyperparameters.get("weight_decay")
            momentum = hyperparameters.get("momentum")
            scheduler_name = hyperparameters.get("scheduler_name")
            
            if scheduler_name is not None:
                net_obj.set_scheduler_name(scheduler_name)

            if lr is not None: 
                net_obj.set_lr(lr)
            else:
                print(Fore.RED+f"Insert information about 'hyperparameters.lr'"+Fore.RESET)
                sys.exit()

            if weight_decay is not None :
                net_obj.set_weight_decay(weight_decay)

            if momentum is not None:
                net_obj.set_momentum(momentum)
        else:
            print(f"Insert information in key 'hyperparameters': lr")
            sys.exit()

        
        criterion = net_obj.get_criterion()
        
        if dataset == "MNIST":
            print(type(net_obj))
            optimizer = net_obj.get_optimizer(net_obj)
        else: # dataset CIFAR10
            optimizer = net_obj.get_optimizer() 
        scheduler = net_obj.get_scheduler()
        
        num_epochs = info_experiment.get("num_epochs", 10)
        batch_size = info_experiment.get("batch_size", 32)
        net_obj.set_batch_size(batch_size)
        legend_for_matrix = info_experiment.get("legend_for_plot")
        if legend_for_matrix is None:
            legend_for_matrix = map_label_to_class_known_reciprocal

        print("Num epochs:", num_epochs, "batch_size:",batch_size)
        print(f"lr {lr}, momentum {momentum}, weight_decay {weight_decay}")


        #---------------------------- create dataset Loader PER DATASET CONCATENATE ------------------------------
        
        #-----Create and Get DATALOADER -----------------------------
        dataset_creator_obj.create_DataLoader("train_known_reciprocal",batch_size)
        dataset_creator_obj.create_DataLoader("validation_known_reciprocal",batch_size)
        
        #-------------- test determinisctoco  
        print(f"TEST DETERMINISTICO SU TRAIN LOADER")
        load_train = dataset_creator_obj.get_datasetLoader("train_known_reciprocal")
        for img, label in load_train:
            print(f"{label}")
            break

        load_train = dataset_creator_obj.get_datasetLoader("train_known_reciprocal")
        for img, label in load_train:
            print(f"{label}")
            break
        
        print(f"TEST DETERMINISTICO SU VALIDATION LOADER")
        load_validation = dataset_creator_obj.get_datasetLoader("validation_known_reciprocal")
        for img, label in load_validation:
            print(f"{label}")
            break
        for img, label in load_validation:
            print(f"{label}")
            break
            
        print(f"TEST DETERMINISTICO SU VALIDATION LOADER da più Batch")
        load_validation = dataset_creator_obj.get_datasetLoader("validation_known_reciprocal")

        selected_batches = []
        for i, batch in enumerate(load_validation):
            if 10 <= i < 15:
                selected_batches.append(batch)
            elif i >= 15:
                break
        i = 0
        for img, label in selected_batches:
            print(Fore.GREEN+f"BATCH [{i}]: {label[0:63]}"+Fore.RESET)
            i = i+1

            
        load_validation = dataset_creator_obj.get_datasetLoader("validation_known_reciprocal")

        selected_batches = []
        for i, batch in enumerate(load_validation):
            if 10 <= i < 15:
                selected_batches.append(batch)
            elif i >= 15:
                break
        i = 0
        for img, label in selected_batches:
            print(Fore.BLUE+f"BATCH [{i}]: {label[0:63]}"+Fore.RESET)
            i = i+1
        #-------------------------------------------------------------------------------------------------
        #########################################################################################
        #---- RUN TRAINING -----------------------------------------------------------------
        print(f"Num epochs {num_epochs}- Batch Size {batch_size}")
        
        run_training(device,  model, optimizer, criterion, scheduler, num_epochs, batch_size, dataset_creator_obj, logdir_root= logdir_root, port_logs=port, type_dataset = "known_reciprocal", experiment = experiment, path_folder = path_folder, path_file_config = path_file_config, continue_training= continue_training)
        
        
        suffix = f"Known_reciprocal_[{experiment_num}]_{dataset}_[TRAIN]"
        idx_known_class = [v for v in known_assign_place.values()]
        print(f"idx_known_class {idx_known_class}")
        name_architecture_obj = net_obj.__class__.__name__
        print(f"Name architecture: {name_architecture_obj}")
        path_model = os.path.join(path_folder_exp, f"model_{experiment}.pth")
        run_visualizing( model, name_architecture_obj, dataset_creator_obj, batch_size, phase = phase, type_dataset = "known_reciprocal",num_classes=num_classes_tot , experiment = experiment, path_model= path_model , path_dest = path_folder_exp, path_file_config = path_file_config, map_label_to_class = legend_for_matrix , idx_known_class = idx_known_class,idx_reciprocal_class=idx_reciprocal_class, outlier=True, suffix=suffix, demo = False )

        #-----------------GET AND SAVE - INFO ABOUT ARCHITECTURE---------------------------------------
        info_architecture = net_obj.get_info_architecture()
        add_update_key(path_file_config, experiment, "info_architecture", info_architecture )

   # automatic testing after the training phase
    #---- RUN TESTING ------------------------------------------------------------------
    if phase == 'test' or phase == 'training':
        path_distribution = None
        path_distribution_validation = None 
        if distribution_test_bool:
            if phase == "training":
                path_distribution = os.path.join(path_folder_exp, f"overlapping_distribution_prob_{phase}_{experiment}_[origin_pop_train_set].csv") # determina la distribuzione del test set o train 
                path_distribution_validation = os.path.join(path_folder_exp, f"overlapping_distribution_prob_{phase}_{experiment}_[origin_pop_valid_set].csv") # determina la distribuzione del test set o train 
            if phase == "test":
                path_distribution = os.path.join(path_folder_exp, f"overlapping_distribution_prob_{phase}_{experiment}_[origin_pop_test_set].csv") # determina la distribuzione del test set o train 


        path_model = os.path.join(path_folder_exp, f"model_{experiment}.pth")
        batch_size = info_experiment.get("batch_size", 32)
        legend_for_matrix = info_experiment.get("legend_for_plot")
        if legend_for_matrix is None:
            legend_for_matrix = map_label_to_class_known_reciprocal
        add_update_key(path_file_config, experiment, "batch_size", batch_size )


        run_testing( model, dataset_creator_obj, batch_size,num_classes_tot, legend_for_matrix, phase = phase, type_dataset = "known_reciprocal", experiment = experiment, path_model= path_model , path_dest = path_folder_exp, path_file_config = path_file_config, distribution_test_bool = distribution_test_bool, path_distribution = path_distribution, path_distribution_validation = path_distribution_validation, classes_index_reciprocal=idx_reciprocal_class )


    if phase == 'visualization':
        path_model = os.path.join(path_folder_exp, f"model_{experiment}.pth")
        batch_size = info_experiment.get("batch_size", 32)
        legend_for_matrix = info_experiment.get("legend_for_plot")
        if legend_for_matrix is None:
            legend_for_matrix = map_label_to_class_known_reciprocal
        name_architecture_obj = net_obj.__class__.__name__
        print(f"Name architecture: {name_architecture_obj}")
        #print(model)
        suffix = f"Known_reciprocal_[{experiment_num}]_{dataset}"
        idx_known_class = [v for v in known_assign_place.values()]
        print(f"idx_known_class {idx_known_class}")

        run_visualizing( model, name_architecture_obj, dataset_creator_obj, batch_size, phase = phase, type_dataset = "known_reciprocal",num_classes=num_classes_tot , experiment = experiment, path_model= path_model , path_dest = path_folder_exp, path_file_config = path_file_config, map_label_to_class = legend_for_matrix , idx_known_class = idx_known_class,idx_reciprocal_class=idx_reciprocal_class, outlier=True, suffix=suffix, demo = False )

    if distribution_test_bool == True and phase != "visualization":
        list_read_path = []
        # read file csv and calculate the threshold min and max of probs
        if phase == "test": 
            list_read_path = [("test",path_distribution)]
        elif phase == "training":
            list_read_path = [("training",path_distribution), ("validation",path_distribution_validation)]
        
        for phase, path_distribution in list_read_path: 
            try:
                df_distribution = pd.read_csv(path_distribution)
                print(f"df_distribution header {phase}")
                print(df_distribution.head())
                distribution_report = { }
                for class_id in range(num_classes_tot):
                    # extract rows with ID_CLASS = class_id 
                    df_class = df_distribution[ df_distribution["ID_CLASS"] == class_id ]
                    # calculate the minimum and maximum for each column related to the probabilities
                    dict_prob = {}
                    for prob_id in range(num_classes_tot):
                        min_prob = df_class[f"prob_{prob_id}"].min() 
                        max_prob = df_class[f"prob_{prob_id}"].max()
                        dict_prob[f"{prob_id}"] = {"min": min_prob, "max": max_prob}

                    distribution_report[f"class_{class_id}"] = dict_prob 
                print(f"Ferma eri {experiment} - distribution_report_set_{phase}")
                # save distribution report in file config
                add_update_key(path_file_config, experiment, f"distribution_report_set_{phase}", distribution_report )
               

            except FileNotFoundError:
                print(Fore.RED+f"Error file is not exist"+Fore.RESET)
                print("eer1")
                sys.exit()
            except pd.errors.ParserError:
                print(Fore.RED+f"Error, the format is not valid"+Fore.RESET)
                print("eer2")
                sys.exit()

            except pd.errors.EmptyDataError:
                print(Fore.RED+f"Error file is empty"+Fore.RESET)
                print("eer3")
                sys.exit()
            except Exception as e:
                print(Fore.RED+f"Error {e}"+Fore.RESET)
                print("eer4")
                sys.exit()
        



    