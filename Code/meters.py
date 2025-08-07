from matplotlib import pyplot as plt
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from utility import add_update_key
import numpy as np
import math
from colorama import Fore
class AverageValueMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0
    
    def add(self, value, num):
        self.sum += value*num
        self.num += num
        
        
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None
        
    def inizializza(self,value,gb):
        self.sum= value*gb
        self.num = gb




class Plotter_Meters():
    def __init__(self):
        #--- ACCURACY ------------------------------
        #-----------------(axis_y)-------------------
        self.array_accuracy_train = []
        self.array_accuracy_validation = [] 

        #---------------- (axis_x) --------------
        # GLOBAL_STEP
        self.array_acc_glb_train = []
        self.array_acc_glb_validation = [] 

        #----LOSS -----------------------------------
        #-----------------(axis_y) ------------------
        self.array_loss_train = []
        self.array_loss_validation = []
        #---------------- (axis_x) --------------
        # GLOBAL_STEP
        self.array_loss_glb_train = []
        self.array_loss_glb_validation = [] 


        #----LOSS_EPOCH------------------------------
        #-----------------(axis_y) ------------------
        self.array_loss_epoch_train = []
        self.array_loss_epoch_validation = []
        #-----------------(axis_x) ------------------
        #EPOCHS
        self.array_epoch_train = []
        self.array_epoch_validation = []

        #-----------------------
        self.time_training = None
        #-----------------------
        self.effective_epoch = 0
        self.num_epochs = 0
        

    def set_timeTraining(self, time):
        self.time_training = time

    def set_effective_epochs(self,effective_epoch,num_epochs):
        self.effective_epoch = effective_epoch
        self.num_epochs = num_epochs


    def append_accuracy(self, mode, acc_value, value_x):
        if mode =='train':
            self.array_accuracy_train.append(acc_value)         # axis_y
            self.array_acc_glb_train.append(value_x)            # axis_x

        elif mode == 'validation':
            self.array_accuracy_validation.append(acc_value)    # axis_y
            self.array_acc_glb_validation.append(value_x)       # axis_x

            

    def append_loss(self, mode, loss_value, value_x):
        if mode =='train':
            self.array_loss_train.append(loss_value)            # axis_y
            self.array_loss_glb_train.append(value_x)           # axis_x

        elif mode == 'validation':
            self.array_loss_validation.append(loss_value)       # axis_y
            self.array_loss_glb_validation.append(value_x)      # axis_x

    def append_loss_epoch(self, mode, loss_value, value_x):
        if mode =='train':
            self.array_loss_epoch_train.append(loss_value)      # axis_y
            self.array_epoch_train.append(value_x)              # axis_x

        elif mode == 'validation':
            self.array_loss_epoch_validation.append(loss_value) # axis_y
            self.array_epoch_validation.append(value_x)         # axis_x


    def save_meters_plot_accuracy(self,  destination, prefix):

        figure = plt.figure(figsize=(12,8))
        plt.plot(self.array_acc_glb_train, self.array_accuracy_train) #The values on x represent the cumulative number of samples visited
        plt.plot(self.array_acc_glb_validation, self.array_accuracy_validation)
        plt.xlabel('samples')
        plt.ylabel('Accuracy')
        plt.title(f'{prefix}: Accuracy ')
        plt.grid()
        plt.legend(['Training','Validation'])
        plt.savefig(os.path.join(destination, f"{prefix}_plot_accuracy.png" ))
        plt.clf() #Clear figure" 
        plt.close(figure)



    def save_meters_plot_loss(self,  destination, prefix):

        figure = plt.figure(figsize=(12,8))
        plt.plot(self.array_loss_glb_train, self.array_loss_train) #The values on x represent the cumulative number of samples visited
        plt.plot(self.array_loss_glb_validation, self.array_loss_validation)
        plt.xlabel('samples')
        plt.ylabel('Loss')
        plt.title(f'{prefix}: Loss ')
        plt.grid()
        plt.legend(['Training','Validation'])
        plt.savefig(os.path.join(destination, f"{prefix}_plot_loss.png" ))
        plt.clf() #Clear figure" (cancella figura)
        plt.close(figure)

    def save_meters_plot_loss_epoch(self, destination, prefix):

        figure = plt.figure(figsize=(12,8))
        plt.plot(self.array_epoch_train, self.array_loss_epoch_train) #The values on x represent the number of epochs
        plt.plot(self.array_epoch_validation, self.array_loss_epoch_validation)
        plt.xlabel('epochs')
        plt.ylabel('Loss for epochs')
        plt.title(f'{prefix}: Loss for epochs')
        plt.grid()
        plt.legend(['Training','Validation'])
        plt.savefig(os.path.join(destination, f"{prefix}_plot_loss_epochs.png" ))
        plt.clf() #Clear figure" (cancella figura)
        plt.close(figure)

    def saveLastPerformance(self,  key, path_config ):

        last_accuracy_train = self.array_accuracy_train[-1]
        last_accuracy_validation = self.array_accuracy_validation[-1]

        last_loss_train = self.array_loss_train[-1]
        last_loss_validation = self.array_loss_validation[-1]

        last_loss_epoch_train = self.array_loss_epoch_train[-1]
        last_loss_epoch_validation = self.array_loss_epoch_validation[-1]


        last_info = { "last_Performance": {"training": {
                                                "accuracy": last_accuracy_train, 
                                                "loss": last_loss_train, 
                                                "loss_epoch": last_loss_epoch_train
                                                },
                                            "validation": {
                                                "accuracy": last_accuracy_validation, 
                                                "loss": last_loss_validation, 
                                                "loss_epoch": last_loss_epoch_validation
                                            }},
                    "time_training": self.time_training,
                    "processing_epochs":{"effective_epoch": self.effective_epoch, "set_epochs":self.num_epochs}
                    }

        add_update_key(path_config,  key, None, last_info)




#------------------- FUNCTION -----------------

def save_confution_matrix_plot(target, prediction, modality, experiment, path_dest, map_label_to_class,observ="", type_task = ""):

    ##-------Confusion matrix -------------------------------------
    #####---- file for confusion_matrix------------
    print(f"Save confution matrix {observ}")

    print(map_label_to_class)
    class_name = [ ]
    #for idx in range(len(map_label_to_class.keys())):
    for idx in map_label_to_class.keys():
        
        name = map_label_to_class.get(idx)
        class_name.append(name)
    
    print(f"Class name = {class_name}")
            
                     
    if observ is None:
        filename_cm = f"cm_[{type_task}]_[{modality}]_{experiment}_standard.png"
    else:
        filename_cm = f"cm_[{type_task}]_[{modality}]_{experiment}_{observ}.png"

    path_confusion_matrix = os.path.join(path_dest,filename_cm)

    title = f"cm_[{modality}]_{experiment}_{observ}"
    accuracy = accuracy_score(target, prediction)
    conf_matrix = confusion_matrix(target, prediction)
    num_classes_MATRIX = conf_matrix.shape[0]
    print(f"num_classes_MATRIX {num_classes_MATRIX}")
    #disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_order_string)
    if len(class_name) < 6:
        fig, ax = plt.subplots(figsize=(9,6))
    elif len(class_name) < 15:
        fig, ax = plt.subplots(figsize=(20,15)) # width, height
    else:
        fig, ax = plt.subplots(figsize=(25,20)) # width, height
        

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_name)
    disp.plot(ax=ax, cmap = "Blues", values_format="d")
    
    # Ruota le etichette delle colonne (asse x)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    if len(class_name) < 6:
        for text in ax.texts:
            text.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=16)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    elif len(class_name) < 16:
        for text in ax.texts:
            text.set_fontsize(19)
        ax.tick_params(axis="both", labelsize=15)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    else: 
        for text in ax.texts:
            text.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=15)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    accuracy_troncata = math.floor(accuracy * 1000) / 1000
    fig.suptitle(f"{title} - Accuracy: {accuracy_troncata:.3f}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    print(f"path to save confusion matrix img {path_confusion_matrix}")
    plt.savefig(path_confusion_matrix)

def save_confution_matrix_plot_perc(target, prediction, modality, experiment, path_dest, map_label_to_class,   observ=None, type_task = ""):
    mask_correct = target == prediction
    print(Fore.CYAN+f"[CM] - tot samples {len(target)}"+Fore.RESET)
    print(Fore.CYAN+f"[CM] - Number of correctly predicted samples {sum(mask_correct)}"+Fore.RESET)

    ##-------Confusion matrix -------------------------------------
    #####---- file per confusion_matrix------------
    print(f"Save confution matrix Perc {observ}")
    print(map_label_to_class)
    class_name = [ ]
    #for idx in range(len(map_label_to_class.keys())):
    for idx in map_label_to_class.keys():
        
        name = map_label_to_class.get(idx)
        class_name.append(name)
    
    print(f"Class name = {class_name}")
            
   
    if observ is  None:
        observ = "standard"
    
    filename_cm = f"cm_[{type_task}]_[{modality}]_{experiment}_perc_{observ}.png"
  


    path_confusion_matrix = os.path.join(path_dest,filename_cm)

    title = f"cm_[{modality}]_{experiment}_{observ}"
    accuracy = accuracy_score(target, prediction)
    print(f"Accuracy {accuracy}")
    conf_matrix = confusion_matrix(target, prediction)

    cm_percent = conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:, np.newaxis] * 100 # percentuali 

    
    if len(class_name) < 5:
        fig, ax = plt.subplots(figsize=(9,6))
    elif len(class_name) < 15:
        fig, ax = plt.subplots(figsize=(20,15)) # width, height
    else:
        fig, ax = plt.subplots(figsize=(25,20)) # width, height

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=class_name)
    disp.plot(ax=ax, cmap = "Blues", values_format=".2f")
    ax.tick_params(axis="both", labelsize=15)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # color bar 
    cbar = ax.images[0].colorbar
    if len(class_name) < 6:
        for text in ax.texts:
            text.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=16)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    elif len(class_name) < 16:
        for text in ax.texts:
            text.set_fontsize(19)
        ax.tick_params(axis="both", labelsize=15)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    else: 
        for text in ax.texts:
            text.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=15)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)

    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    accuracy_troncata = math.floor(accuracy * 1000) / 1000
    fig.suptitle(f"{title} - Accuracy: {accuracy_troncata:.3f}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path_confusion_matrix)



def demo_confution_matrix_plot_openset_perc(target, prediction, modality, experiment, path_dest, legend_matrix, idx_reciprocal, idx_unknown = None, observ=None):
    print(f"Demo confusion matrix - idx_unknown : {idx_unknown} - legend_matrix {legend_matrix}")
    ##-------Confusion matrix -------------------------------------
    # create a mask to identify samples with target belonging to reciprocal_class
    mask_target_reciprocal  = np.isin(target, idx_reciprocal) 

    # create a mask to identify samples with predictions belonging to reciprocal_class
    mask_predicted_reciprocal = np.isin(prediction, idx_reciprocal)

 

    # Create a mask with True where both mask_target_reciprocal and mask_predicted_reciprocal are True
    # Replace target values where mask_target_reciprocal is True with idx_unknownl 
    # Replace predicted values where mask_predicted_reciprocal is True with idx_unknown
    target[mask_target_reciprocal] = idx_unknown 
    prediction[mask_predicted_reciprocal] = idx_unknown 

    print(f"[demo CM] Number of samples with TARGET unknown after replacement: {sum(np.isin(target, idx_unknown))}")
    print(f"[demo CM] Number of samples with PREDICTED unknown after replacement: {sum(np.isin(prediction, idx_unknown))}")
    



    #####---- file per confusion_matrix------------
    print("Save confution matrix Perc DEMO OPEN SET")
   
    if observ is not None:
        filename_cm = f"demo_openset_[{modality}]_{experiment}_perc_{observ}.png"
    else:
        filename_cm = f"demo_openset_[{modality}]_{experiment}_perc.png"


    path_confusion_matrix = os.path.join(path_dest,filename_cm)

    title = f"OpenSet_[{modality}]_{experiment}"
    accuracy = accuracy_score(target, prediction)
    conf_matrix = confusion_matrix(target, prediction)

    cm_percent = conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:, np.newaxis] * 100 # percentuali 

    #disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_order_string)
    if len(legend_matrix) < 5:
        fig, ax = plt.subplots(figsize=(9,6))
    elif len(legend_matrix) < 15:
        fig, ax = plt.subplots(figsize=(20,15)) # width, height
    else:
        fig, ax = plt.subplots(figsize=(25,20)) ## width, height


    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=legend_matrix)
    disp.plot(ax=ax, cmap = "Blues", values_format=".2f")
    ax.tick_params(axis="both", labelsize=15)
    # Ruota le etichette delle colonne (asse x)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # color bar 
    cbar = ax.images[0].colorbar
    if len(legend_matrix) < 6:
        for text in ax.texts:
            text.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=16)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    else:
        for text in ax.texts:
            text.set_fontsize(18)
        ax.tick_params(axis="both", labelsize=15)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    
    accuracy_troncata = math.floor(accuracy * 1000) / 1000
    fig.suptitle(f"Demo - {title} - Accuracy: {accuracy_troncata:.3f}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path_confusion_matrix)

def demo_confution_matrix_plot_openset(target, prediction, modality, experiment, path_dest, legend_matrix, idx_reciprocal, idx_unknown = None, observ=None):

    ##-------Confusion matrix -------------------------------------
    # Create a mask to identify samples with target of reciprocal_class 
    mask_target_reciprocal  = np.isin(target, idx_reciprocal) 

    # Create a mask to identify samples with predictions of reciprocal_class
    mask_predicted_reciprocal = np.isin(prediction, idx_reciprocal)

    # Create a mask with True where both target and predicted are reciprocal_class 
    #mask_true_reciprocal = mask_target_reciprocal & mask_predicted_reciprocal 

    # Replace target and predicted values corresponding to mask_target_reciprocal or mask_predicted_reciprocal 
    # with the unknown label index, for both targets and predictions
    target[mask_target_reciprocal] = idx_unknown 
    prediction[mask_predicted_reciprocal] = idx_unknown 

    mask_correct = target == prediction 
    print(Fore.CYAN+f"[CM - DEMO] FINAL CM - TOT samples {len(target)}"+Fore.RESET)
    print(Fore.CYAN+f"[CM - DEMO] FINAL CM - Number of correctly predicted samples {sum(mask_correct)}"+Fore.RESET)

    #####---- file per confusion_matrix------------
    print("Save confution matrix  DEMO OPEN SET")
   
    if observ is not None:
        filename_cm = f"demo_openset_[{modality}]_{experiment}_{observ}.png"
    else:
        filename_cm = f"demo_openset_[{modality}]_{experiment}.png"


    path_confusion_matrix = os.path.join(path_dest,filename_cm)

    title = f"OpenSet_[{modality}]_{experiment}"
    accuracy = accuracy_score(target, prediction)
    conf_matrix = confusion_matrix(target, prediction)

    #cm_percent = conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:, np.newaxis] * 100 # percentuali 

    #disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_order_string)
    if len(legend_matrix) < 6:
        fig, ax = plt.subplots(figsize=(9,6))
    else:
        fig, ax = plt.subplots(figsize=(20,15)) # width, height

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=legend_matrix)
    disp.plot(ax=ax, cmap = "Blues", values_format="d")
    ax.tick_params(axis="both", labelsize=15)
    # Rotate the column labels (x-axis)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # color bar 
    cbar = ax.images[0].colorbar
    if len(legend_matrix) < 6:
        for text in ax.texts:
            text.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=16)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    else:
        for text in ax.texts:
            text.set_fontsize(18)
        ax.tick_params(axis="both", labelsize=15)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    accuracy_troncata = math.floor(accuracy * 1000) / 1000
    fig.suptitle(f"Demo - {title} - Accuracy: {accuracy_troncata:.3f}", fontsize=18)
    plt.savefig(path_confusion_matrix)



def calculate_metrics_report( target, prediction, map_label ):

    print(f"Etichette uniche target {np.unique(target)}")

    ##----- Calculate accuracy and derivers metrics---------------------------------------
    accuracy = accuracy_score(target, prediction)
    precision_for_class = precision_score(target, prediction, average=None, zero_division=0)
    precision_macro = precision_score(target, prediction, average='macro', zero_division=0)

    recall_for_class = recall_score(target, prediction, average=None, zero_division=0)
    recall_macro = recall_score(target, prediction, average='macro',zero_division=0)

    f1_for_class = f1_score(target, prediction, average=None, zero_division=0)
    f1_macro = f1_score(target, prediction, average='macro',zero_division=0)

    cm = confusion_matrix(target, prediction) 
    #print(f"cm {cm}")
    # Calculation of accuracy per class in percentage
    accuracy_perc_for_class = {} 
    class_accuracy = (cm.diagonal()/ cm.sum(axis=1))*100 
    print(f"Accuracy % per classe - {class_accuracy}")
    print(f"MAp_label {map_label} - nuemro della classi {len(class_accuracy)} ")
    for label, acc in zip(map_label.keys(),class_accuracy):
        #print(f"Tipo i {type(label)} - i:{label}")
        if isinstance(list(map_label.keys())[0], str):
            print(f"Accuracy for class_name {map_label.get(str(label))} - label {label}: {acc:.2f}%") 
        else:
            print(f"Accuracy for class_name {map_label.get(label)} - label {label}: {acc:.2f}%") 
        accuracy_perc_for_class[label] = round(acc, 2)

    if map_label is not None:
        label = sorted(list(map_label.keys()))
        print("Label", label)
        precision_for_class = {k:v for k, v in zip(label, precision_for_class) }
        recall_for_class = {k:v for k, v in zip(label, recall_for_class) }
        f1_for_class = {k:v for k, v in zip(label, f1_for_class) }
    else:
        precision_for_class.tolist()
        precision_for_class.tolist()
        f1_for_class.tolist()





    print(f"accuracy_score: {accuracy}")

    print(f"precision_for_class: {precision_for_class}")
    print(f"precision_macro: {precision_macro}")

    print(f"recall_score: {recall_for_class}")
    print(f"recall_macro:{recall_macro}")

    print(f"f1_for_class: {f1_for_class}")
    print(f"f1_macro: {f1_macro}")

    dict_metrics = {}
    dict_metrics[f"accuracy"] = accuracy
    dict_metrics[f"precision_for_class"] = precision_for_class
    dict_metrics[f"precision_macro"] = precision_macro
    dict_metrics[f"recall_for_class"] = precision_for_class
    dict_metrics[f"recall_macro"] = recall_macro
    dict_metrics[f"f1_for_class"] = f1_for_class
    dict_metrics[f"f1_macro"] = f1_macro
    dict_metrics[f"accuracy_perc_for_class"] = accuracy_perc_for_class

    return dict_metrics
