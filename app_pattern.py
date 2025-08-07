import os
import json
import sys
import argparse
from utility import *
from colorama import Fore, Back, Style
import threading
import torch
from Pattern_Creator import Pattern_Creator
from Manager_Networks import Manager_Networks
from Transformer import Transformer
import pickle
import configparser

root = os.getcwd()
folder_results_dataset_known = "Results_dataset_known" #
folder_config_pattern = "Config_pattern"
logdir_root = "logs_fitness_pattern"
port_logs = 6007
path_home_dataset = None



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


seed = 123 

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Pattern Generation with Evolutive algorithm NEAT- CPP')
    parser.add_argument('--ref_experiment', type=int, choices={1,2}, required= True, help="Indicate the experiment number, to refer to the DNN trained on known classes ")
    parser.add_argument('--config_pattern_file', type=str, help="JSON file to configure pattern generation, filename with prefix [config_]", required=True)
    parser.add_argument('--genoma_best', action="store_true" )
    parser.add_argument('--num_run', type=int )
    parser.add_argument('--num_best', type=int )
    parser.add_argument('--fitness', type=str )
    parser.add_argument('--fine_tuning', action="store_true" )

    args = parser.parse_args()
    #----------------------
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_home_dataset = config["absolute_path"]["datasets"]
    print(f"Absolute path for datasets {path_home_dataset}")

    #----------PARSE ARGS -----------------------------------------------
    fitness_value = None
    experiment_num = args.ref_experiment
    file_config = args.config_pattern_file
    ref_experiment = f"experiment_[{str(experiment_num)}]"
    generate_with_best_genoma = args.genoma_best
    if generate_with_best_genoma:
        num_best = args.num_best
        num_run_prev = args.num_run
        fitness_value = args.fitness
        if num_best is None or num_run_prev is None or fitness_value is None :
            print("insert num_run and num_best and fitness_value")
            sys.exit()
    
    fine_tuning = args.fine_tuning
    if fine_tuning:
        num_best = args.num_best
        num_run_prev = args.num_run
        fitness_value = args.fitness
        if num_best is None or num_run_prev is None or fitness_value is None :
            print("insert num_run and num_best and fitness_value")
            sys.exit()


    #------------- check if filename begin with config_ ------------------
    if file_config.startswith("config_"):
        print("Ok prefix")
    else:
        print(Fore.RED+ f"Error, your configuration file must has prefix [config_]"+Fore.RESET)
        sys.exit()

    
    #--------------check if folder Config_pattern exists-------------------
    path_folder_config_pattern = os.path.join(root, folder_config_pattern )
    if not check_exists(path_folder_config_pattern):
        create_Folder(path_folder_config_pattern)
        folder_ref_experiment = f"ref_{ref_experiment}"
        path_folder_config_pattern_ref_experiment = os.path.join(path_folder_config_pattern, folder_ref_experiment)
        create_Folder(path_folder_config_pattern_ref_experiment)
        print(Fore.RED+ f"Compile file of configuration [{file_config}] in Config_pattern.{folder_ref_experiment} folder"+Fore.RESET)
        path_file_config = os.path.join(path_folder_config_pattern_ref_experiment, file_config )
        create_file_json_pattern(path_file_config)
        sys.exit()
    else:
        folder_ref_experiment = f"ref_{ref_experiment}"
        path_folder_config_pattern_ref_experiment = os.path.join(path_folder_config_pattern, folder_ref_experiment)
        create_Folder(path_folder_config_pattern_ref_experiment)
        #---------- check if file_config_pattern exists --------------------
        path_file_config = os.path.join(path_folder_config_pattern_ref_experiment, file_config )
        if not check_exists(path_file_config):
            print(Fore.RED+ f"The configuration file [{file_config}] in Config_pattern.{folder_ref_experiment} folder is missing"+Fore.RESET)
            path_file_config = os.path.join(path_folder_config_pattern_ref_experiment, file_config )
            print(Fore.MAGENTA+ f"Creating of the configuration file [{file_config}] in Config_pattern.{folder_ref_experiment}"+Fore.RESET)
            create_file_json_pattern(path_file_config)
            print(f"{path_file_config}")
            print(Fore.RED+ f"Compile file of configuration [{file_config}] in Config_pattern.{folder_ref_experiment} folder"+Fore.RESET)
            sys.exit()

    #--------- LOAD FILE of CONFIGURATION ----------------------

    try:
        with open(path_file_config, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"File {file_config} is not present, insert it in folder [{folder_config_pattern}]" )
        create_file_json_pattern(path_file_config)
        sys.exit()
    except Exception as e:
        print(f"Error {e}")
        sys.exit()
    
        #-------- READ key [ref_experiment] in  pattern CONFIGURATION------------

    if config.get("ref_experiment"):
        exp_number = config["ref_experiment"].get("number")
        if ref_experiment == exp_number:
            name_architecture = config["ref_experiment"].get("architecture_obj")
            model_filename = config["ref_experiment"].get("filename_model", f"model_{ref_experiment}.pth")
        else:
            print(Fore.RED+f"The numbers of ref_experiment do not match. Check the ref_experiment.number key in the configuration file with the one given in --ref_experiment "+Fore.RESET)
            sys.exit()
        if config["ref_experiment"].get("task_classification"):
            num_classes = config["ref_experiment"]["task_classification"].get("num_classes")
            if num_classes is None:
                print(Fore.RED+f"Insert value in key in ref_experiment.task_classification.num_classes "+Fore.RESET)
                sys.exit()
        else:
            print(Fore.RED+f"Insert key  [ref_experiment.task_classification.num_classes] "+Fore.RESET)
            sys.exit()

        if config["ref_experiment"].get("src_dataset"):
            src_dataset = config["ref_experiment"].get("src_dataset")
        else:
            print(Fore.RED+f"Insert key [ref_experiment.src_dataset] in config file "+Fore.RESET)
            sys.exit()
        if config["ref_experiment"].get("model_type"):
            model_type =  config["ref_experiment"].get("model_type")
        else:
            print(Fore.RED+f"Insert key [ref_experiment.model_type] in config file "+Fore.RESET)
            sys.exit()

    else:
        print(Fore.RED+f"Insert key [ref_experiment] in config file "+Fore.RESET)
        sys.exit()


    print(f"Exp_number {exp_number}, Num classes {num_classes}")

    






    #-----------set Folder and Path for Model DNN ------------------------

    path_results = os.path.join(root,folder_results_dataset_known)
    path_ref_experiment = os.path.join(path_results, ref_experiment )

    name_file_model = f"model_{ref_experiment}.pth"
    path_model = os.path.join(path_ref_experiment, name_file_model)
    

    #----------check if the model exists -------------------------------
    if check_exists(path_model):
        print(Fore.GREEN+f"Model {name_file_model} exists"+ Fore.RESET)
    else:
        print(Fore.RED+ f"Model {name_file_model} is not present"+Fore.RESET)
        sys.exit()
    

    #-------- READ key [reciprocal_pattern] in  pattern CONFIGURATION -------------

    if config.get("reciprocal_pattern"):
        reciprocal_pattern = config.get("reciprocal_pattern")
    else: 
        print(Fore.RED+ f"Define the key [reciprocal_pattern] in the pattern configuration file"+Fore.RESET)
        sys.exit()
    
   
    recip_class = None
    array_all_class_name = []
    array_all_class_number = []
    print(Fore.CYAN+f"Recipocal patterns are:"+Fore.RESET)
    for item in reciprocal_pattern:   #es  [{"0": "bookcase"},{ "1": "gorilla"}]
        for label, class_name in item.items():
            print(f"  Label: {label}, Class_name: {class_name}")
            recip_class = int(label)
            array_all_class_name.append(class_name)
            array_all_class_number.append(int(label))
    
 


    
    
      

    #---------------------MANAGER NETWORKS ------------------------------------------
    
    manager_networks_obj = Manager_Networks()
    architecture_obj = manager_networks_obj.get_architecture(name_architecture, num_classes)
    print(type(architecture_obj))#<class 'Manager_Networks.ResNet18_IMAGENET'>
    size_img = architecture_obj.get_size_img_input()
    if src_dataset == "IMAGENET":
        net_architecture = architecture_obj.get_model() 
    elif src_dataset == "MNIST":
        net_architecture = architecture_obj

    print(type(net_architecture)) # <class 'torchvision.models.resnet.ResNet'>
    
    prefix = extract_string(file_config,"config_")
    print(f"Prefix {prefix}")
    filename_report_net = f"neat_report_[{prefix}].csv"

    dest = os.path.join(path_folder_config_pattern_ref_experiment, filename_report_net)
    path_best_genoma = os.path.join(path_folder_config_pattern_ref_experiment, f"model_best_genoma_{ref_experiment}_{prefix}.pkl")
    #-------------------- TRANSFORMER --------------------------------------------
    transformer_obj = Transformer(None, None)
    inf_norm =transformer_obj.get_normalized(src_dataset)
    transformer_obj.set_for_dataset_type("pretrained")
    transforms_test = transformer_obj.get_transforms(src_dataset, model_type, False, False)

   #-------------------- PATTER CREATOR --------------------------------------------
    pattern_creator_obj = Pattern_Creator()
    pattern_creator_obj.set_net_architecture(net_architecture, size_img, path_model, path_best_genoma)
    pattern_creator_obj.set_normalize(inf_norm)
    pattern_creator_obj.set_transformer(transforms_test)
    pattern_creator_obj.set_path_home_datasets(path_home_dataset)
    #pattern_creator_obj.set_constrain(constrain)
    pattern_creator_obj.set_folder_config(path_folder_config_pattern_ref_experiment)
    file_name_csv = f"{src_dataset}_known_filename_img.csv"
    path_csv = os.path.join(path_home_dataset,file_name_csv)
    pattern_creator_obj.set_path_csv_filename(path_csv)
    pattern_creator_obj.all_reciprocal_class = array_all_class_name
    pattern_creator_obj.all_reciprocal_class_number = array_all_class_number
    #------------------------------------------------------------------------------

    
    
    #########################################################################
    
    pattern_creator_obj.set_filename_report(dest)

    if not check_exists(dest):
        write_intestation_csv(dest,prefix,num_classes)

    #----------------------------------ACTIVATE TENSOR BOARD - PORT 6007--- LOGdor_root = logs_pattern-----------------------------------
    port_logs = 6020 

    logdir_root = f"{logdir_root}_{src_dataset}"
    tb_thread = threading.Thread( target = start_tensorboard, args = (logdir_root, port_logs) )
    tb_thread.start()

    pattern_creator_obj.set_logdir(logdir_root)


    #-------------PATH to SAVE PATTERN AND REPORT CSV -----------------------------------------
    folder_pattern = src_dataset[0].upper() + src_dataset[1:].lower() + '_patterns'
    path_folder_pattern = os.path.join(path_home_dataset, folder_pattern)
    create_Folder(path_folder_pattern)
    folder_exp = f"pattern_{ref_experiment}"
    path = os.path.join(path_home_dataset,folder_pattern)
    folder_exp = os.path.join(path,folder_exp )
    create_Folder(folder_exp)
    path_prefix = os.path.join(folder_exp,prefix )
    create_Folder(path_prefix)

    num_run = find_num_run(path_file_config)
    pattern_creator_obj.set_num_run(num_run)
    pattern_creator_obj.set_path_folder_experiment(folder_exp)

    #----------------------RUNNING_GENERATIVE_RECIPROCAL_PATTERN------of ONE class------------------------------------
    print(f"src_datasset {src_dataset}")
    
    if src_dataset=="MNIST":
        pattern_RGB = False
        pattern_creator_obj.set_bool_RGB(False)
        pattern_creator_obj.sorgente_folder = "MNIST_photos"
    elif src_dataset == "IMAGENET":
        pattern_RGB = True 
        pattern_creator_obj.set_bool_RGB(True)
        pattern_creator_obj.sorgente_folder = "Imagenet_photos"


    #----------------------RUNNING_GENERATIVE_RECIPROCAL_PATTERN------of ALL class- Neutral Class-----------------------------------
    if (len(reciprocal_pattern) == num_classes):
        pattern_creator_obj.set_num_classes_tot(num_classes)
        pattern_creator_obj.reciprocal_class = "all"
        print("Generation of pattern of Neutral class")
        if src_dataset=="MNIST":
            pattern_RGB = False
            pattern_creator_obj.set_bool_RGB(False)
        elif src_dataset == "IMAGENET":
            pattern_RGB = True 
            pattern_creator_obj.set_bool_RGB(True)
      
        if generate_with_best_genoma == False:
            pattern_creator_obj.run_generative_reciprocal_patterns(path_prefix, num_run, all_reciprocal_pattern = True)
        else:
            filename_genoma_best = f"run_[{num_run_prev}]_best_[{num_best}]_[{str(fitness_value)}].pkl"
            print(f"file genoma best {filename_genoma_best}")
            
           
            path_model_best = path_folder_config_pattern_ref_experiment
            
            print(f"reciprocal all {path_prefix}, {path_model_best}, {path_model_best}")
            
            pattern_creator_obj.run_generate_with_best_genoma(path_prefix, num_run, path_model_best,filename_genoma_best, reciprocal_pattern= None, all_reciprocal_pattern=True )
    



    i = pattern_creator_obj.counter_best +1
    
    pattern_creator_obj.save_report(num_run)                              
    pattern_creator_obj.save_id_genoma()
    #---------------------------------------------
    info_final = pattern_creator_obj.info_generation()
    add_update_key(path_file_config, "run", num_run, info_final)

