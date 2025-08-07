import torch 
import numpy
import os
import json
import sys
import argparse
import torch.nn as nn
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
folder_result = "Results_dataset_known"
path_home_dataset = None
print(torch.__version__)
print(torchvision.__version__)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed):
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
    parser.add_argument('--distribution', action="store_true", help="Collect distribution features for classes")
    parser.add_argument('--phase', type=str, choices = {'training', 'test', "miss_classification"}, required= True)

    parser.add_argument('--continue_training',  action="store_true")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--id",
        type=int,
        help="Image ID of the dataset_known test. Required in --phase 'miss_classification",
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

    if (phase == 'test' or phase == 'training') and distribution_test_bool == True:
        distribution_test_bool = True
    else: 
        distribution_test_bool = False

    num_classes = None
    
    #--------- CREATE or check FOLDER-------------------
    path_folder = os.path.join(root, folder_result)
    create_Folder(path_folder)  

    #--------- CHECK EXISTS FOLDER EXPERIMENT -------------------
    path_folder_exp = os.path.join(path_folder, experiment )
    create_Folder(path_folder_exp)

    #--------- LOAD FILE of CONFIGURATION ----------------------
    path_file_config = os.path.join(path_folder_exp,file_config)

    try:
        with open(path_file_config, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"File config.json is not present, insert it in folder [{folder_result}/{experiment}]" )
        create_file_json_known(path_file_config)
        sys.exit()
    except Exception as e:
        print(f"Error {e}")
        sys.exit()

    #--------- READ FILE OF CONFIGURATION -----------------
    folder_photos = config.get("folder_photos")
    known_name_classes = config.get("known_name_classes")
    path = os.path.join(path_home_dataset,folder_photos)
    dataset = config.get("src_dataset")
    
    info_experiment = config.get(experiment)
    if info_experiment is not None:
        balanced = info_experiment.get("balanced")
        network = info_experiment.get("network").get("architecture") 
        pretrained = info_experiment.get("network").get("pretrained")
        dropout = info_experiment.get("network").get("dropout", None)
        print(type(pretrained), pretrained)
        if network is  None or pretrained is  None:
            print(f"Insert information under key 'experiment_[{str(experiment_num)}]':'network.achitecture , network.pretrained'")
            sys.exit()
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
        num_classes = len(known_name_classes)
        print(f"Num classes: {num_classes}")
    if dataset is None:
        print(f"Insert information about 'src_dataset'")
        sys.exit()
    if balanced is None:
        print(f"Insert information about 'balanced' in key 'experiment[{experiment_num}]'")
        sys.exit()



    #-------------------------------------------------
    observ="standard"
    if phase == 'miss_classification':
        path_file_miss = os.path.join(path_folder_exp, f"miss_classified_{experiment}_{observ}.json")
        show_missclassification(path_file_miss, experiment, miss_class_id_or_all)
        sys.exit()
        
    #----------- initialize Dataset_Creator------------
    print(f"Dove salvare le immagini scaricate da dataset { path}")
    dataset_creator_obj = Dataset_Creator(seed, generator)
    dataset_creator_obj.setFolder_Photos_Known(path)

    for n in known_name_classes:
        print(f"known class name {n}")
        dataset_creator_obj.add_known_class_name(n)

    #---------- Tranformer-------------------------
    transformer_obj = Transformer(seed, generator)
    transformer_obj.set_for_dataset_type("pretrained") # in questo modo utilizza media e dev di imagenet 
    transforms_all = transformer_obj.get_transforms( dataset, network, bool_all= True, bool_train=None) # traforms for all dataset
    transforms_train = transformer_obj.get_transforms( dataset, network, bool_all= None, bool_train=True) # trasforms for training set
    transforms_test = transformer_obj.get_transforms( dataset, network, bool_all= None, bool_train=False) # trasformazione for test set
    #print(transforms_train)
    ############------------------------------------##########
    ############       Create Dataset KNOWN         ##########
    ############------------------------------------##########
    if dataset == "IMAGENET":
        dataset_creator_obj.create_dataset_from_ImageFolder(type_task = "known", validation_size=0.15, test_size=0.15, transforms_all = transforms_all, transforms_train= transforms_train, transforms_test=transforms_test) # effettua anche lo splits
        logdir_root = "logs_known_IMAGENET"
        port = 6006
        
    elif dataset == "MNIST":
        Folder_Photos = f"{dataset}_photos"
        path_Folder_Photos = os.path.join(path_home_dataset, Folder_Photos)
        create_Folder(path)
        logdir_root = "logs_known_MNIST"
        port = 6008
        path_dataset_download = os.path.join(path_home_dataset, "MNIST_data")
        dataset_creator_obj.create_dataset_from_Datasets(root=path_dataset_download, type_task ="known", type_dataset = dataset, validation_size=0.15, transforms_all = transforms_all, transforms_train = transforms_train, transforms_test = transforms_test)
    


    
    for type_t in ["overall","train", "validation", "test"]:
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
    for idx in [10, 20, 50]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("validation_known", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[validation_known] - {path_img} - {label} - {class_name}"+Fore.RESET)
    for idx in [40, 70, 110, 1000]:
        path_img, label, class_name = dataset_creator_obj.get_sample_info("test_known", idx)
        if path_img is not None:
            print(Fore.CYAN+f"[test_known] - {path_img} - {label} - {class_name}"+Fore.RESET)
    
    #########################################################################
    file_csv_path_img = f"{dataset}_known_filename_img.csv"
    df_train = dataset_creator_obj.get_dict_path_img("train_known")
    df_validation = dataset_creator_obj.get_dict_path_img("validation_known")
    df_test =dataset_creator_obj.get_dict_path_img("test_known")

    path_csv_path_img = os.path.join(path_home_dataset, file_csv_path_img)
    df_union = pd.concat([df_train, df_validation, df_test], ignore_index= True)
    df_union.to_csv(path_csv_path_img, index=False)
    #######################################################################




    ##----------- NETWORK  --------------------------------------------
    manager_network = Manager_Networks()
    if pretrained == True:
        print("Pretrained TRUE")
        net_obj = manager_network.get_network_pretrained(network, dataset, len(known_name_classes), dropout)
        print("Nome criterion class",net_obj.get_criterion_name())
        model = net_obj.get_model()

    elif pretrained == False:
        print("Pretrained FALSE")
        net_obj = manager_network.get_network_no_pretrained(network, dataset, len(known_name_classes),dropout)
        print("Nome criterion class",net_obj.get_criterion_name())
        model = net_obj
    #print(type(net_obj))
    #------------ PHASE TRAINING ------------------------------------------------------
    if phase == 'training':

        #----- get info about hyperparameters e settings from file of configuration ----------
        hyperparameters = info_experiment.get("hyperparameters")
        if hyperparameters is not None:
            lr = hyperparameters.get("lr")
            weight_decay =  hyperparameters.get("weight_decay")
            momentum = hyperparameters.get("momentum")


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

    criterion_type = None
    info_architecture = info_experiment.get("info_architecture")
    if info_architecture is not None:
        criterion = info_architecture.get("criterion")
        if criterion is None: 
            criterion = info_experiment.get("criterion")
        if criterion is not None:
            criterion_type = criterion.get("type")
            print(f"Criterion function {criterion_type}")
            if criterion_type is not None:
                if criterion_type == "JSD_Loss":
                    net_obj.set_criterion(JSD_Loss())
                elif criterion_type == "CrossEntropyLoss":
                    net_obj.set_criterion( nn.CrossEntropyLoss(reduction='mean'))



    if phase == "training":        
        scheduler_info = info_experiment.get("scheduler")
        if scheduler_info is not None:
            scheduler_name = scheduler_info.get("type")
            print(scheduler_name)
            
            if scheduler_name is not False:
                net_obj.set_scheduler_name(scheduler_name)
                print(f"Scheduler name {scheduler_name}")

        
        criterion = net_obj.get_criterion()
        if dataset == "MNIST":
            print(type(net_obj))
            optimizer = net_obj.get_optimizer(net_obj)
        else:
            optimizer = net_obj.get_optimizer() 
        
        scheduler = net_obj.get_scheduler()
        
        num_epochs = info_experiment.get("num_epochs", 10)
        batch_size = info_experiment.get("batch_size", 32)
        net_obj.set_batch_size(batch_size) 

        print("Num epochs:", num_epochs, "batch_size:",batch_size, "Scheduler type" , type(scheduler) )
        print(f"lr {lr}, momentum {momentum}, weight_decay {weight_decay}")
        

        #---------------------------- create dataset Loader ------------------------------
        
        #-----Create and Get DATALOADER -----------------------------
        dataset_creator_obj.create_DataLoader("train_known",batch_size)
        dataset_creator_obj.create_DataLoader("validation_known",batch_size)

        #-------------- deterministic test 
        print(f" DETERMINISTIC TEST ON TRAIN LOADER")
        load_train = dataset_creator_obj.get_datasetLoader("train_known")
        for img, label in load_train:
            print(f"{label[10:20]}")
            break

        load_train = dataset_creator_obj.get_datasetLoader("train_known")
        for img, label in load_train:
            print(f"{label[10:20]}")
            break
        
        print(f" DETERMINISTIC TEST ON VALIDATION LOADER")
        load_validation = dataset_creator_obj.get_datasetLoader("validation_known")
        for img, label in load_validation:
            print(f"{label[40:50]}")
            break

        load_validation = dataset_creator_obj.get_datasetLoader("validation_known")
        for img, label in load_validation:
            print(f"{label[40:50]}")
            break
        

    

        #---- RUN TRAINING -----------------------------------------------------------------

        run_training( device, model, optimizer, criterion, scheduler, num_epochs, batch_size, dataset_creator_obj, logdir_root= logdir_root, port_logs=port, type_dataset = "known", experiment = experiment, path_folder = path_folder, path_file_config = path_file_config, continue_training= continue_training)

        #-----------------GET AND SAVE - INFO ABOUT ARCHITECTURE---------------------------------------
        info_architecture = net_obj.get_info_architecture()
        add_update_key(path_file_config, experiment, "info_architecture", info_architecture )
        
    #  automatic testing on training set
    #---- RUN TESTING ------------------------------------------------------------------
    if phase == 'test' or phase == 'training':
        path_distribution = None
        path_distribution_validation = None 
        if distribution_test_bool:
            if phase == "training":
                path_distribution = os.path.join(path_folder_exp, f"distribution_prob_{phase}_{experiment}_[origin_pop_train_set].csv") # determina la distribuzione del test set o train 
                path_distribution_validation = os.path.join(path_folder_exp, f"distribution_prob_{phase}_{experiment}_[origin_pop_valid_set].csv") # determina la distribuzione del test set o train 
            if phase == "test":
                path_distribution = os.path.join(path_folder_exp, f"distribution_prob_{phase}_{experiment}_[origin_pop_test_set].csv") # determina la distribuzione del test set o train 

        path_model = os.path.join(path_folder_exp, f"model_{experiment}.pth")
        batch_size = info_experiment.get("batch_size", 32)
        add_update_key(path_file_config, experiment, "batch_size", batch_size )
        run_testing( model, dataset_creator_obj, batch_size,num_classes, map_label_to_class, phase = phase, type_dataset = "known", experiment = experiment, path_model= path_model , path_dest = path_folder_exp, path_file_config = path_file_config, distribution_test_bool = distribution_test_bool, path_distribution = path_distribution, path_distribution_validation=path_distribution_validation )

        suffix = f"Known_[{experiment_num}]_{dataset}_[{phase}]"
        phase_v = f'visualization_{phase}'
        path_model = os.path.join(path_folder_exp, f"model_{experiment}.pth")
        batch_size = info_experiment.get("batch_size", 32)
        map_label_to_class = dataset_creator_obj.get_known_map_label_to_name()
        name_architecture_obj = net_obj.__class__.__name__
        print(f"Name architecture: {name_architecture_obj}")
        #print(model)
        idx_known_class = [int(key) for key in map_label_to_class.keys()]
        print(f"idx_known_class {idx_known_class}")

        run_visualizing( model, name_architecture_obj, dataset_creator_obj, batch_size, phase = phase_v, type_dataset = "known",num_classes=num_classes,  experiment = experiment, path_model= path_model , path_dest = path_folder_exp, path_file_config = path_file_config, map_label_to_class = map_label_to_class, idx_known_class=idx_known_class, suffix=suffix  )
        



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
                distribution_report = { }
                for class_id in range(num_classes):
                    # estrarre le sole righe con ID_CLASS = class_id 
                    df_class = df_distribution[ df_distribution["ID_CLASS"] == class_id ]
                    # calcolare il minimo e il max per ciascuna colonna reative alle probabilita 
                    dict_prob = {}
                    for prob_id in range(num_classes):
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


