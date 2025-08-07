from sklearn.metrics import accuracy_score
import os
import torch
from utility import check_exists, add_update_key, get_key, write_init_file_json
import sys
from meters import calculate_metrics_report, save_confution_matrix_plot, save_confution_matrix_plot_perc, demo_confution_matrix_plot_openset_perc, demo_confution_matrix_plot_openset
import numpy as np
import torch.nn.functional as F
import pandas as pd
from colorama import Fore
from utility import *

def test_classifier( model, loader, path_model, distribution_test_bool, num_classes, path_distribution=None, read_distribution_min=None,  read_distribution_max=None,observ = None, classes_index_reciprocal=None, replace_idx_unknown=8, read_distance = None, vector_dev_std = None ):
    #print(f"Path model !!!!!!!{path_model}")
    print(f"MIN_DISTRIBUTION {read_distribution_min}")
    #print(f"DISTANCE {read_distance}")
    #print(f"Dev_std {vector_dev_std}")
    print(f"observ {observ}")


    if check_exists(path_model):
        print(f"Model is present")
    else:
        print(f"Model is not present, change model")
        sys.exit()
   
    #------------ LOAD MODEL ------------------------------------------    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    checkpoint = torch.load(path_model, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        model.load_state_dict(checkpoint)

    
    model.eval()

    contatore = 0 
    predictions = []
    labels = []

    if distribution_test_bool:
        columns = ["ID_CLASS"]
        column_probs = [f"prob_{i}" for i in range(num_classes)] # key for dict

        columns.extend(column_probs)

        data_ditribution = []

        data_ditribution_with_overlap_reciprocal = []

    print("START Execution TESTING ...")
    
    with torch.no_grad():
        for data, target in loader:
           
            x = data.to(device)
            y = target.to(device)
            output_model = model(x)
            #print(output)
            contatore = contatore + len(y)

            if isinstance(output_model, tuple):
                
                features, logit = output_model
                features = features.to('cpu').numpy()
            else:
                logit = output_model


            output = F.softmax(logit, dim=1)  #class predictions (probability vector)

            
            pred = output.to('cpu').max(1)[1].numpy() # among the C probabilities, find the max -> output (max_val, index) -> take index == class_num
            label = y.to('cpu').numpy()


            
            if observ is not None or observ != "standard":
                if observ == "threshold":
                    if read_distribution_min is not None:
                        
                        distrib_min_prob = np.array(read_distribution_min) 


                        max_prob = output.to('cpu').max(1)[0].numpy()
                        indice = output.to('cpu').max(1)[1].numpy()


                        min_prob_for_batch = distrib_min_prob[indice]

                        
                            
                        # copy
                        adjusted_classes = indice.copy()

                        # mask for class 0 e 3
                        mask_low_classes = ~np.isin(indice, classes_index_reciprocal)

                        # condition: class 0–3 and probability < threshold
                        mask_replace = (mask_low_classes) & (max_prob < min_prob_for_batch)

                        # replace with index unknown
                        adjusted_classes[mask_replace] = replace_idx_unknown

                    
                        pred = adjusted_classes
                        
                
            predictions.extend(list(pred))
            labels.extend(list(label))


            if distribution_test_bool == True:
                #print(f"distribution_test_bool {distribution_test_bool}")
                label_real = np.array(label)
                label_pred = np.array(pred)
               

                prob_output = output.to('cpu').numpy()
                if classes_index_reciprocal is  None:
                    #print("Computer distribuion without reciprocal")
                    mask = label_pred == label_real # [True False True False False]
                    for idx, v in enumerate(mask):
                       
                        if v == True:
                            class_real = label_real[idx]
                            prob_classes = list(prob_output[idx]) # valori

                            dict_obj = dict(zip(column_probs, prob_classes))

                            dict_obj["ID_CLASS"] = class_real

                            data_ditribution.append(dict_obj)
                
                #############################################
                #versione 2 calcolo di didtributione con reciproci overlap 
                elif classes_index_reciprocal is not None:
                    #print("Computer distribuion with reciprocal")
                    mask_1 = label_pred == label_real 
                
                    real_mask_reciprocal = np.isin(label_real, classes_index_reciprocal)
                    pred_mask_reciprocal = np.isin(label_pred, classes_index_reciprocal)

                    mask_2 = real_mask_reciprocal & pred_mask_reciprocal 

                    mask_final = mask_1 | mask_2 
                    for idx, v in enumerate(mask_final):
                        #print(idx)
                        if v == True:
                            class_pred = label_pred[idx]
                            prob_classes = list(prob_output[idx]) #
                            
                            dict_obj = dict(zip(column_probs, prob_classes))

                            dict_obj["ID_CLASS"] = class_pred


                            # add row in dataframe 
                            data_ditribution.append(dict_obj)




    if distribution_test_bool == True:
        # create datframe from dict
        df_distribution = pd.DataFrame(data_ditribution)
        df_distribution = df_distribution[columns]
        # save as file csv 
        print(Fore.GREEN+f"Save file csv that describes distribution of classes {path_distribution}"+Fore.RESET)
        df_distribution.to_csv(path_distribution, index=False)
        print(df_distribution)
        ##################################
        # crate datframe from dict
        df_distribution = pd.DataFrame(data_ditribution)
        df_distribution = df_distribution[columns]
        path_distribution_2 = f"{path_distribution}"
        print(Fore.GREEN+f"Save file csv that describes distribution of classes {path_distribution_2}"+Fore.RESET)
        df_distribution.to_csv(path_distribution_2, index=False)
        print(df_distribution)

        print(f"End phase TESTING -  {contatore}")

    return np.array(predictions), np.array(labels)




def run_testing( model, dataset_creator_obj,batch_size,num_classes, map_label_to_class, phase , type_dataset , experiment , path_model, path_dest, path_file_config, distribution_test_bool, path_distribution=None, read_distribution=None, read_distance = None,distr_max_unknown=None,  observ="standard", classes_index_reciprocal=None, replace_idx_unknown=None,path_distribution_validation=None, vector_dev_std = None):

    print(f"Execution TEST during [{phase}] phase...")
    print(f"MAP legend LEGEND {map_label_to_class}")
    print(Fore.RED+f"compute distribution? {distribution_test_bool}"+Fore.RESET)


    #------------ LOAD DATASET and create or Get DataLoader---------------------------------------
    datasets_loader = {}
    mode = []

    if phase == 'training':
        if type_dataset == 'known':
            print("Obtain known training set - dataloader")
            train = dataset_creator_obj.get_datasetLoader("train_known") # dataset training splitted per batch

            print("Obtain known validation set - dataloader")
            validation = dataset_creator_obj.get_datasetLoader("validation_known") 
        
        elif type_dataset == 'known_reciprocal':
            print("!Obtain known_reciprocal - training set - dataloader")
            train = dataset_creator_obj.get_datasetLoader("train_known_reciprocal") # dataset training splitted per batch

            print("Obtain known_reciprocal -validation set - dataloader")
            validation = dataset_creator_obj.get_datasetLoader("validation_known_reciprocal") 
        


        if train is None or validation is None:
            print("Error! DatasetLoaders are not present!")
            sys.exit()

        datasets_loader["training"] = train
        datasets_loader["validation"] = validation
        mode.append("training")
        mode.append("validation")

    elif phase == 'test' or phase == "testing":
        if type_dataset == 'known':
            dataset_creator_obj.create_DataLoader("test_known", batch_size)
            test = dataset_creator_obj.get_datasetLoader("test_known")
        
        elif type_dataset == 'known_reciprocal':
            dataset_creator_obj.create_DataLoader("test_known_reciprocal", batch_size)
            test = dataset_creator_obj.get_datasetLoader("test_known_reciprocal")

        elif type_dataset == 'test_known_reciprocal_unknown':
            print("!Obtain open_set_testing - test set - dataloader")
            test = dataset_creator_obj.get_datasetLoader("test_known_reciprocal_unknown")

        if test is None:
            print("Error! DatasetLoader is not present")
            sys.exit()
        
        print(f"TEST DETERMINISTICO SU TEST LOADER in testing")
        for img, label in test:
            print(f"{label}")
            break

        
        for img, label in test:
            print(f"{label}")
            break
        
        datasets_loader["test"] = test
        mode.append("test")


    
    for modality in mode:
        predictions = []
        labels = []
        distrib_eval = distribution_test_bool

        print(f"Mode [{modality}]")
        if path_distribution_validation is not None:
            if modality == "validation":
                path_distribution = path_distribution_validation 
        #------------- RUN TESTING -----------------------------------------------------------
        prediction, target = test_classifier( model, datasets_loader[modality], path_model, distrib_eval, num_classes, path_distribution, read_distribution, distr_max_unknown,observ, classes_index_reciprocal=classes_index_reciprocal, replace_idx_unknown=replace_idx_unknown, read_distance = read_distance, vector_dev_std = vector_dev_std)

        #print(f"[testing] num of elements - len of array of target {len(target)}")
        corretti = sum(t == p for t, p in zip(target, prediction))
        #print(f"[testing] num correct elements {corretti}")
        
        #------------- CONFUTION MATRIX --------------------------------------------------
        save_confution_matrix_plot(target, prediction, modality, experiment, path_dest, map_label_to_class,observ, type_task = type_dataset)
        save_confution_matrix_plot_perc(target, prediction, modality, experiment, path_dest, map_label_to_class, observ, type_task = type_dataset) 
        map_label = get_key(path_file_config, f"{type_dataset}", "label_to_className")
        if map_label is None:
            map_label = map_label_to_class
        
        #print(f"MAP_LABEL {map_label}")
        #sys.exit()
        dict_metrics = calculate_metrics_report( target, prediction, map_label)
        add_update_key(path_file_config, experiment, f"report_{modality}_{observ}", dict_metrics )

        if type_dataset == "test_known_reciprocal_unknown":
            legend_matrix_demo  = dataset_creator_obj.known_class_name.copy()
            legend_matrix_demo.append("unknown")  # names of knwon + "unknown"
            idx_reciprocal = list(dataset_creator_obj.reciprocal_map_label_to_name.keys())
            idx_unknown = len(dataset_creator_obj.known_class_name) + len(dataset_creator_obj.reciprocal_class_name)
            print(f"Info DEMO - legend {legend_matrix_demo}  - unknown {idx_unknown} - idx_reciprocal {idx_reciprocal}")
            target_copy = target.copy()
            prediction_copy = prediction.copy()
            demo_confution_matrix_plot_openset_perc(target_copy, prediction_copy, modality, experiment, path_dest, legend_matrix_demo, idx_reciprocal, idx_unknown , observ)
            demo_confution_matrix_plot_openset(target_copy, prediction_copy, modality, experiment, path_dest, legend_matrix_demo, idx_reciprocal, idx_unknown , observ)
            map_label_demo = dataset_creator_obj.known_map_label_to_name
            map_label_demo[idx_unknown] = "unknown" 
            dict_metrics_demo = calculate_metrics_report( target_copy, prediction_copy, map_label_demo)
            add_update_key(path_file_config, experiment, f"DEMO_report_{modality}_{observ}", dict_metrics_demo )
        #--------- REPORT -----------------------------------------------------------
       


        if modality == 'test':
            #----------- save the path of misclassified images  -----------------------------
            missclassified = {}
            for i, _ in enumerate (prediction):
                if prediction[i] != target[i]:
                    if type_dataset!= "test_known_reciprocal_unknown":
                        path_image, sample_target, class_name = dataset_creator_obj.get_sample_info(f"test_{type_dataset}", i)
                    else:
                        path_image, sample_target, class_name = dataset_creator_obj.get_sample_info("test_known_reciprocal_unknown",i)
                       
                    if all(isinstance(k, int) for k in map_label.keys()):
                        name_class_pred = map_label.get(prediction[i])
                    else:
                        name_class_pred = map_label.get(str(prediction[i]))
                        
                    missclassified[f"{i}"] = {"path_image": path_image, "target": {int(sample_target): class_name}, "predicted_label": {int(prediction[i]):name_class_pred}}
                  

            path_file_miss = os.path.join(path_dest, f"miss_classified_{experiment}_{observ}.json")
            if check_exists(path_file_miss) == False:
                print(f"Create file miss_classified_{experiment}_{observ}.json")
                init_json = {"description":f" List of paths of misclassified images - {experiment} "}
                write_init_file_json(path_file_miss, init_json)

            add_update_key(path_file_miss, experiment, None, missclassified)






















