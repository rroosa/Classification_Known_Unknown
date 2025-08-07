from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from utility import check_exists
import matplotlib.patches as mpatches  
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import torch.nn.functional as F
from colorama import Fore
from utility import*
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def define_min_max(path_distribution, num_classes):
    try:
            df_distribution = pd.read_csv(path_distribution)
            distribution_report = { }
            for class_id in range(num_classes):
                # extract rows with ID_CLASS = class_id 
                df_class = df_distribution[ df_distribution["ID_CLASS"] == class_id ]
                # Calculate the minimum and maximum for each column related to the probabilities 
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
    


def compute_distribution(label_real, label_pred, prob_output, num_classes, path_distribution, outlier_idx = None):
    print(f"Num classses {num_classes}")
    
    data_ditribution = []
    columns = ["ID_CLASS"]
    column_probs = [f"prob_{i}" for i in range(num_classes)] # keys for dict
    columns.extend(column_probs)
    count_tot = 0
    count_process = 0
    mask = label_pred == label_real
    #print(f" index di outliner {outlier_idx}+ {len(outlier_idx)}")
    for idx, v in enumerate(mask):
   
        if v == True:
            if outlier_idx is None or (outlier_idx is not None and idx not in outlier_idx):
                class_real = label_real[idx]
                prob_classes = prob_output[idx] # valori
                print(f"prob_output {prob_output.shape}")
                print(f"column_probs {len(column_probs)}")
                print(f"prob_class {prob_classes.shape}")
                print(f"type prob_class {type(prob_classes)}")
                print(f"type column_probs {type(column_probs)}")

                # Create a dictionary from a list of keys and a list of values.
                dict_obj = dict(zip(column_probs, list(prob_classes)))
                
                dict_obj["ID_CLASS"] = class_real
                # Add the key ID_CLASS: 0"

                # add the row to the dataframe 
                data_ditribution.append(dict_obj)
                count_process =  count_process +1
            
            count_tot = count_tot +1 

    
    # crate datframe from dict
    df_distribution = pd.DataFrame(data_ditribution)
    df_distribution = df_distribution[columns]
    # save as csv file
    print(Fore.GREEN+f"Save file csv that describes distribution of classes {path_distribution}"+Fore.RESET)
    df_distribution.to_csv(path_distribution, index=False)
    #print(df_distribution)
    
    print(f"Num. processed sample {count_process}")



def extractor_features( name_architecture, model, loader, path_model, list_min=None, classes_index_reciprocal=None, replace_idx_unknown= None, observ=None, list_distance = None): 

    if check_exists(path_model):
        print(f"Model is present")
    else:
        print(f"Model is not present, change model")
        sys.exit()
    
    print(f"MIN_DISTRIBUTION {list_min}")
    #print(f"LIST DISTANCE  {list_distance}")

    print(f"Observ  {observ}")

    #------------ LOAD MODEL ------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    checkpoint = torch.load(path_model, map_location=device)

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

    else:
        model.load_state_dict(checkpoint)
    model.eval()

    list_features = []
    list_target = []
    list_predicted_labels = []
    list_probs = []
    list_distrib_vector = []
    print("START Execution Visualization ...")
    contatore = 0
    with torch.no_grad():
        for samples, label in loader:
            batch_size = label.shape[0]
            images = samples.to(device)
            target = label.to(device)
            contatore = contatore + len(target)
            features = model(images)    # class predictions (vector of numbers)
            if isinstance(features, tuple):
                featues_emb , distrib_prob = features
                features = distrib_prob
           
            output_prob = F.softmax(features, dim=1)  # class predictions (probability vector)
            max_prob, indice = output_prob.max(1)
            max_prob = max_prob.cpu().numpy()

            num_classes = output_prob.shape[1]
            predicted_labels = output_prob.to('cpu').max(1)[1].numpy()

 
            """
            if observ == "mean_distance_mean_centroide" or observ == "median_distance_mean_centroide" or observ == "mean_distance_median_centroide" or observ == "median_distance_median_centroide":
                
                centroide_for_class, limite_distance_for_class = zip(*list_distance) # array di tuple spacchettare in due tuple separate
                centroide_for_class = list(centroide_for_class) # trasforma tuple in lista vettori_centroide
                limite_distance_for_class = list(limite_distance_for_class)

                limite_distance_for_class = [ distance +0.02 for distance in limite_distance_for_class]

                #print(centroide_for_class)
                #print(limite_distance_for_class)
                limite_distance_for_class = np.array(limite_distance_for_class)
                centroide_for_class = np.array(centroide_for_class)
                indice = output_prob.to('cpu').max(1)[1].numpy()
                output = output_prob.to('cpu').numpy()
                limite_distance_for_batch = limite_distance_for_class[indice]
                centroide_for_batch = centroide_for_class[indice]

                #print(f"Centroide_for_batch: {centroide_for_batch}")
                #print(f"Limite_distance_for_batch: {limite_distance_for_batch}")
                # Copia iniziale
                adjusted_classes = indice.copy()

                distance_sample_centroide_for_batch = np.linalg.norm(output - centroide_for_batch, axis=1)

                
                # Costruisci maschera per classi tra 0 e 3
                mask_low_classes = ~np.isin(indice, classes_index_reciprocal)
                mask_replace = (mask_low_classes) & (distance_sample_centroide_for_batch > limite_distance_for_batch)

                adjusted_classes[mask_replace] = replace_idx_unknown
                predicted_labels = adjusted_classes
            """

            if observ == "threshold"  and list_min is not None:
                #print("Applied the threshold")
                max_prob, indice = output_prob.max(1)  # max_prob is a tensor containing the maximum probabilities  
                                                        # predicted_class is a tensor containing the indices of the predicted classes
                distrib_min_prob = np.array(list_min) 
                # Convert tensor in NumPy 
                max_prob = max_prob.cpu().numpy()
                indice = indice.cpu().numpy()
                #print(f"Reference probability" {distrib_min_prob}")

                #print(f"Predicted class {indice}")
                #print(f"Real class {target}")
                #print(f"Max prob  {max_prob}")
                min_prob_for_batch = distrib_min_prob[indice]
                
                adjusted_classes = indice.copy()
                mask_low_classes = ~np.isin(indice, classes_index_reciprocal)
                
                mask_replace = (mask_low_classes) & (max_prob < min_prob_for_batch)
                # Replaced with index unknown
                adjusted_classes[mask_replace] = replace_idx_unknown

                #print(f"New predicted classes {adjusted_classes}")
                predicted_labels = adjusted_classes
            """
            elif observ == "cross":
                    
                #print("Entra in cross")
                centroide_for_class, limite_distance_for_class = zip(*list_distance) # array di tuple spacchettare in due tuple separate
                centroide_for_class = list(centroide_for_class) # trasforma tuple in lista vettori_centroide
                limite_distance_for_class = list(limite_distance_for_class)
                offset = 0.02
                limite_distance_for_class = [ distance + offset  for distance in   limite_distance_for_class ]

                limite_distance_for_class = np.array(limite_distance_for_class)

                centroide_for_class = np.array(centroide_for_class) # 8*8
                #print(f"Limite DISTANCE {limite_distance_for_class.shape}, {limite_distance_for_class}")
                #print(f"CENTROIDE for classe {centroide_for_class}")


                out_prob = output_prob.to('cpu').numpy() # 32*8
                indice = output_prob.to('cpu').max(1)[1].numpy()

                adjusted_classes = indice.copy()

                diff = out_prob[:, np.newaxis, :] - centroide_for_class[np.newaxis, :, :]  # shape: (32, 8, 8)
                dists = np.linalg.norm(diff, axis=2)  # shape: (32, 8)

                #print(dists.shape)  # (32, 8)

                #--- calcolo confronto puntuale distanza > raggio
                mask_2d = dists > limite_distance_for_class

                mask_all_true_per_sample = np.all(mask_2d, axis=1)  # shape: (32,) # controllo per sample se tutti i confronti sono vere

                #print(mask_all_true_per_sample.shape)  # (32,)
                #print(mask_all_true_per_sample)   # ottengo per campione True o False 
                # se è true allora significa che è unknown 
                #adjusted_classes[mask_all_true_per_sample] = replace_idx_unknown
                #####------------------------------------
                # Ottieni gli indici dei valori False per ciascuna riga- individuando dunque l'indice del cluster il cui campione è dentro
                # Lista degli indici dei `False` per ogni riga, oppure [-1] se tutti sono True
                indici_cluster_dentro = [
                    np.where(~riga)[0] if not np.all(riga) else np.array([-1])
                    for riga in mask_2d
                ]
                #print(len(indici_cluster_dentro))
                #print(indici_cluster_dentro)
                #print("Step2")
                # Step 2: Costruisci nuovo array: 8 se [-1], altrimenti il primo indice , 
                adjusted_classes = np.array([
                    replace_idx_unknown if (len(array_indici) == 1 and array_indici[0] == -1) else array_indici[0]
                    for array_indici in indici_cluster_dentro
                ])

                #print(f"Classi aggiistate \n {adjusted_classes}")
                predicted_labels = adjusted_classes
            """

            list_predicted_labels.extend(list(predicted_labels)) 
            

            label = target.to('cpu').numpy()
            list_target.extend(list(label))
           
            list_distrib_vector.append(output_prob.cpu().numpy())     
            list_features.append(features.cpu().numpy())                      
            list_probs.append(max_prob)                         




        all_predicted_labels = np.array(list_predicted_labels)
        all_target = np.array(list_target)



        all_distrib_vector = np.concatenate(list_distrib_vector, axis=0)
        all_probs = np.concatenate(list_probs, axis=0)
        all_features = np.concatenate(list_features,axis=0)


        print(f"Shape features_map of all sample {all_features.shape}")
        print(f"Shape target of all sample {all_target.shape}")
        print(f"Shape predicted_ labels of all sample {all_predicted_labels.shape}")
        print(f"Target {all_probs.shape}")
        print(f"Counter {contatore}")


    print(Fore.GREEN+f"[extractor_features] Number of elements – length of the labels array {len(all_target)}"+Fore.RESET)
    corretti = sum(t == p for t, p in zip(all_target, all_predicted_labels))
    print(Fore.GREEN+f"[extractor_features] Number of correct elements {corretti}"+Fore.RESET)
    return all_distrib_vector, all_target, all_predicted_labels, all_features



def viewer_3D( probs_array, target_array, predicted_array, map_label_to_class, path_exp , observ="", idx_known_class =None, idx_reciprocal_class= None, idx_unknown= None, outlier=None, suffix="", bool_openset_testing=False, demo=False):
    print("Elaborate Plotly express - TSNE-3D ...")

    num_classes = len( list(map_label_to_class.keys() )) # num_classi
    print(f"Num Classes in tot {num_classes}, map label to all class {map_label_to_class}, Idx class know {idx_known_class}, Idx reciprocal class {idx_reciprocal_class}, Idx_unknown {idx_unknown}")
    
    unici = list(set(target_array))
    print(f"Unique Targert  {unici}")
    outlier_indices = None
    print(Fore.CYAN+f"[Viewer] Number of elements – length of the labels array {len(target_array)}"+Fore.RESET)
    corretti = sum(t == p for t, p in zip(target_array, predicted_array))
    print(Fore.CYAN+f"[Viewer] Number of correct elements {corretti}"+Fore.RESET)

    # ---------  t-SNE -----------------------------------------
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    features_3d = tsne.fit_transform(probs_array)  # Output in 3D

    if demo:
        print(f"Number of samples initially with target unknown: {sum(np.isin(target_array, idx_unknown))}")
        print(f"Number of samples initially with predicted unknown: {sum(np.isin(predicted_array, idx_unknown))}")

        mask_target_reciprocal = np.isin(target_array, idx_reciprocal_class)
        mask_predicted_reciprocal = np.isin(predicted_array, idx_reciprocal_class)
       
        print(f"Replace RECIP -> indx_unknown {idx_unknown} in target for {sum(mask_target_reciprocal)} samples -  and replace RECIP -> indx_unknown {idx_unknown} in predicted for {sum(mask_predicted_reciprocal)} samples ")
        
        target_array[mask_target_reciprocal] = idx_unknown
        predicted_array[mask_predicted_reciprocal] = idx_unknown

        print(f"Number of samples, after replacement, with target unknown: {sum(np.isin(target_array, idx_unknown))}")
        print(f"Number of samples, after replacement, with predicted unknown: {sum(np.isin(predicted_array, idx_unknown))}")



    #---------------------------------
    # Creazione del grafico 3D scatter
    df = pd.DataFrame({
        "x": features_3d[:, 0],
        "y": features_3d[:, 1],
        "z": features_3d[:, 2],
        "Predicted label": predicted_array,
        "Target class": target_array
    })

    color_known            = ['rgb(20, 90, 50)','rgb(255, 136, 2)', 'rgb(74, 35, 90)', 'rgb(21, 67, 96)', 'rgba(250, 25, 172, 0.58)' , 'rgb(255,0,0)', 'rgb(255,215,0)', ]
                                # green                # orange        # purple          # blue               # fuxsia                #red              #gold                                           

    color_reciprocal_class = ['rgb(130, 224, 170)', 'rgb(248, 196, 113 )', 'rgb(195, 155, 211)', 'rgb(133, 193, 233)','rgba(250, 25, 172, 0.39)' ,  'rgb(255,99,71)',  'rgb(240,230,140)']
                                # light green         # light orange           # light purple          # light blue         # light fuchsia               # light red    # khaki                
    
    color_reciprocal_all   = [ 'rgb(0,255,255)'] # acqua marine
                                
    color_unknown =     ['rgb(204, 209, 209)'] # light grey

    print(f"Id known classes {idx_known_class} - Id reciprocal classes {idx_reciprocal_class} - Idx unknown {idx_unknown}")
    
    # matching colors
    color = []
    # 1) case with only known classes 
    if idx_reciprocal_class is None:
        print(Fore.MAGENTA+"Case 1 -  only  known classes"+Fore.RESET)
        # select the first k colors for the known classes only
        for i in range(len(idx_known_class)):
            color.append(color_known[i])
    
    # 2) case known classes == reciprocal classes 
    if idx_reciprocal_class is not None and demo == False:
        
        if len(idx_reciprocal_class) == len(idx_known_class): # caso numero known = reciproci
            print(Fore.MAGENTA+"Case 2 -  known classes == RECIPROCAL classes"+Fore.RESET)
            if max(idx_known_class) < min(idx_reciprocal_class):
                print(f"Case 2.1 - known indices BEFORE reciprocal indices{max(idx_known_class)} <{ min(idx_reciprocal_class)}")
                # This means that the reciprocal indices are positioned after the known indices
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])
                for j in range(len(idx_reciprocal_class)):
                    color.append(color_reciprocal_class[j])
            else:
                print(f"Case 2.2 - known indices alternated with reciprocal indices{idx_known_class} {idx_reciprocal_class}")
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])
                    color.append(color_reciprocal_class[i])
                    

        elif len(idx_reciprocal_class) > len(idx_known_class):  # case  number of known classes < reciproci = reci + all
            print(Fore.MAGENTA+"Case 3 - known classes < RECIPROCHE = recip_i + recip_all"+Fore.RESET)
            if max(idx_known_class) < min(idx_reciprocal_class):
                # This means that the reciprocal indices are positioned after the known ones. 
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])
                for j in range(len(idx_reciprocal_class)-1):
                    color.append(color_reciprocal_class[j])

                color.append(color_reciprocal_all[0]) # add color of reciproco_all 
        
        elif len(idx_reciprocal_class) < len(idx_known_class): # case known > reciprocal
            print(Fore.MAGENTA+f"Case 4 -  known classes > RECIPROCHE {len(idx_reciprocal_class)}"+Fore.RESET)
            if len(idx_reciprocal_class) == 1:
                 
                print(Fore.MAGENTA+"Case 4.1 - RECIPROCAL classes = 1 -> recip_all"+Fore.RESET)
                # case reciprocal is only one classe reciprocal_all 
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])

                color.append(color_reciprocal_all[0])

            elif len(idx_reciprocal_class) > 1:
                print(Fore.MAGENTA+"Case 4.2 -  known classes == 2 -> brother of some class"+Fore.RESET)
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])
                for i in range(len(idx_reciprocal_class)):
                    color.append(color_reciprocal_class[i])


    elif demo==True:
        for i in range(len(idx_known_class)):
            color.append(color_known[i])

    
    #  Check if the unknown class is present, indicating open-set testing; add the color for unknown.
    if bool_openset_testing and demo == False: 
        if len(idx_known_class) + len(idx_reciprocal_class) < len(map_label_to_class):
            print(" UNKNWON class is present - add the color ")
            color.append(color_unknown[0])
    
    elif bool_openset_testing:
        color.append(color_unknown[0])


    print(f"Selected color array – length {len(color)}:  {color}")



    if idx_reciprocal_class is not None and bool_openset_testing:
        unknown_class = idx_unknown
        print(f"Correct column Target class - real label for unknown class= {unknown_class}")
        df['Correct_Target'] = np.where( (df['Target class'] == unknown_class) & (df['Predicted label'].isin(idx_reciprocal_class)),True, False )

        count = (df['Predicted label'] == unknown_class).sum()
        print(f"Number of elements with predicted unknown: {count}")
        
        count_target = (df['Correct_Target']).sum()
        print(f"Number of elements with Correct_Target: {count_target}")
    
    print(f"length of array of color {len(color)} == lengtg of keys {len(map_label_to_class)}")
    
    colors_MAP = {}
    init = 0
    i = 0
    bar = []
    dist = []
    fraz = 1.0/(len(map_label_to_class.keys())-1)
    labels = list(map_label_to_class.keys())
    
    for i, label in enumerate(labels):
        colors_MAP[int(label)] = color[i]
        bar.append([init,color[i]])
        dist.append(init)
        init = init + fraz
        i += 1
    
    dist[-1] = 1.0 
    bar[-1] = [1.0, color[-1]]

    print("Color Map \n",colors_MAP)
    print("Bar \n",bar)
    print("Dist \n",dist)


    count1 = (df['Predicted label'] == df['Target class']).sum()
    print(f"Number of elements with (predicted = real): {count1}")

    
    if df.get('Correct_Target') is  None:
        df["Symbol"] = df.apply( lambda row: "circle" if row["Target class"] == row["Predicted label"]  else "diamond", axis = 1) # passare riga per riga (axis=1)

    if df.get('Correct_Target') is not None:
        df["Symbol"] = df.apply( lambda row: "circle" if row["Target class"] == row["Predicted label"] or row["Correct_Target"] else "diamond", axis = 1) # passare riga per riga (axis=1)

        countcorrect = (df['Correct_Target']).sum()
        print("colonna Correct_Target è presente")
        print(f"Numeri di elemeti con Correct_Target: {countcorrect}")
        
    

    countcicle = (df['Symbol'] == "circle").sum()
    print(Fore.GREEN+f"Nember of elements with circle in Symbol {countcicle}"+Fore.RESET)
        
        
    
    df["hover_text"] = df.apply(lambda row: f"Target class: {row['Target class']}<br>Predicted label: {row['Predicted label']}", axis=1)  # striga da stampare nell'etichetta tooltip
    
    # Creazione della colonna per le etichette  della legenda
    df["Legend Symbol"] = df["Symbol"].map({"circle": "Hit Classification", "diamond": "Miss Classification"})

    fig = go.Figure()

    # Add the trace for points with "Hit Classification" (circle))
    fig.add_trace(go.Scatter3d(
        x=df[df["Symbol"] == "circle"]["x"],
        y=df[df["Symbol"] == "circle"]["y"],
        z=df[df["Symbol"] == "circle"]["z"],
        mode="markers",
        marker=dict(
            symbol="circle", size=8, # correct predicted = circle
            
            color= df[df["Symbol"] == "circle"]["Target class"].astype(int),  # color based on target lable
            cmin= 0,
            cmax=len(map_label_to_class)-1,
            
            colorscale = bar,
            opacity=0.5,
            
            colorbar=dict(
                title="Target Classes",
                
                tickvals = list(range(len(map_label_to_class))),
                ticktext=[f"{key} : {map_label_to_class[key]}" for key in map_label_to_class.keys()], # label the ticks with the class names
                len=1.0  #  lenght della barra
            ),
            showscale=True,  # show the colorbar
           

        ),
        name="Hit Classification",  #Name of lagend
        hovertext=df[df["Symbol"] == "circle"]["hover_text"],
        hoverinfo="text"
    ))

    # Add the trace for points with "Miss Classification" (diamond)
    fig.add_trace(go.Scatter3d(
        x=df[df["Symbol"] == "diamond"]["x"],
        y=df[df["Symbol"] == "diamond"]["y"],
        z=df[df["Symbol"] == "diamond"]["z"],
        mode="markers",
        marker=dict(
            symbol="diamond", size=8, 

            color= df[df["Symbol"] == "diamond"]["Target class"].astype(int),
            cmin= 0,
            cmax=len(map_label_to_class)-1,
            colorscale = bar,
            
            opacity=0.7,
            line=dict(width=3, color="red"), 
            showscale=False,  # deactivate colorbar,
           
        ),
        name="Miss Classification",  # name in legenda
        hovertext=df[df["Symbol"] == "diamond"]["hover_text"],
        hoverinfo="text"
    ))

    # customized legend
    
    fig.update_layout(
        title=f"t-SNE 3D -Predicted Classes \n {suffix}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        legend=dict(
            x=0,  # [0-1] horizontal posiztion: right -> left
            y=1,  # [0-1] vertical position: bottom -> top
            xanchor="left",  # "left", "center", "right"
            yanchor="top",  # "top", "middle", "bottom"
        ),
        hovermode="closest"
    )
    if demo:
        path_dest = os.path.join(path_exp,f"Demo_plotly_tsn_3D_prob_{suffix}.html")
    else:
        path_dest = os.path.join(path_exp,f"plotly_tsn_3D_prob_{suffix}.html")
    
    print(f"Save plot in path {path_dest}")
    fig.write_html(path_dest)
   

def scatter_3D( features_array, target_array, predicted_array, map_label_to_class, path_exp, observ=""):
    
    print("Elaborate Scatter TSNE-3D ...")

    num_classes = len( list(map_label_to_class.keys() )) # num_classi
    print(f"Num Classes {num_classes}")

    # ---------  t-SNE -----------
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    features_3d = tsne.fit_transform(features_array)  # Output in 3D

    # --------- PLOT  POINTS IN 3D 
    plt.ion()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')


    # Define a colormap and assign a color to each class
    colors = plt.cm.plasma(np.linspace(0, 1, num_classes))  # Create  distint color

    # create dict to map classes -> colors
    color_dict = {i: colors[i] for i in range(num_classes)}

    # Scatter plot with colors based on real class
    scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=[color_dict[label] for label in target_array],  alpha=0.7)

    # title and axes
    ax.set_title(f"Visualization 3D with t-SNE - Feature Space \n {suffix}")
    ax.set_xlabel("Dim-1 x")
    ax.set_ylabel("Dim-2 y")
    ax.set_zlabel("Dim-3 z")

    # ===== ADD LEGEND =====
    legend_patches = [mpatches.Patch(color=color_dict[int(key)], label=f"{key}:{map_label_to_class[str(key)]}") for key in map_label_to_class.keys()]
    ax.legend(handles=legend_patches, title="Class known", loc="upper right")


    if observ is None:
        path_dest = os.path.join(path_exp,f"scatter_tsn_3D_prob.png")
    else:
        path_dest = os.path.join(path_exp,f"scatter_tsn_3D_prob_{observ}.png")
    plt.savefig(path_dest)
    plt.show()
    plt.close()




def viewer_2D( probs_array, target_array, predicted_array, map_label_to_class, path_exp , observ="", idx_known_class =None, idx_reciprocal_class= None,idx_unknown =None, outlier=None, suffix="", bool_openset_testing=False, demo=False):
    print("Elaborate Plotly express - TSNE-2D ...")

    num_classes = len( list(map_label_to_class.keys() )) # num_classi
    print(f"Num Classes in tot {num_classes}, map label to all class {map_label_to_class}, Idx class know {idx_known_class}, Idx reciprocal class {idx_reciprocal_class}")
    
    unici = list(set(target_array))
    print(f"Targert unici {unici}")
    outlier_indices = None

    # ---------  t-SNE -----------------------------------------
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(probs_array)  # Output in 3D

    if demo:
        mask_target_reciprocal = np.isin(target_array, idx_reciprocal_class)
        mask_predicted_reciprocal = np.isin(predicted_array, idx_reciprocal_class)
        indx_unknown = len(idx_known_class) + len(idx_reciprocal_class)
        target_array[mask_target_reciprocal] = indx_unknown
        predicted_array[mask_predicted_reciprocal] = indx_unknown



    #---------------------------------
    # Creazione del grafico 2D scatter
    df = pd.DataFrame({
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "Predicted label": predicted_array,
        "Target class": target_array
    })

    color_known            = ['rgb(20, 90, 50)','rgb(255, 136, 2)', 'rgb(74, 35, 90)', 'rgb(21, 67, 96)', 'rgba(250, 25, 172, 0.58)' , 'rgb(255,0,0)', 'rgb(255,215,0)', ]
                                # verde                # arancione        # viola          # blu                # fuxsia                #rosso              #gold                                           

    color_reciprocal_class = ['rgb(130, 224, 170)', 'rgb(248, 196, 113 )', 'rgb(195, 155, 211)', 'rgb(133, 193, 233)','rgba(250, 25, 172, 0.39)' ,  'rgb(255,99,71)',  'rgb(240,230,140)']
                                # verde chiaro         # arancione chiaro     # viola chiaro        # blu chiaro           # fuxiasia chiaro           #rosso chiaro         # khaki                  
    
    color_reciprocal_all   = [ 'rgb(0,255,255)'] # acqua marine
                                
    color_unknown =     ['rgb(204, 209, 209)'] # grigio chiaro

    print(f"[2D] Id known classes {idx_known_class} - Id reciprocal classes {idx_reciprocal_class} - Id unknown {idx_unknown}")
    
    # effettua l'abbinamento dei colori 
    color = []
    # 1) caso di sole classi note 
    if idx_reciprocal_class is None:
        print(Fore.MAGENTA+"Case 1 -  only known classes"+Fore.RESET)
        # seleziona i primi k colori per le sole classi note 
        for i in range(len(idx_known_class)):
            color.append(color_known[i])
    
    # 2) caso classi note == reciproci 
    if idx_reciprocal_class is not None and demo == False:
        print(Fore.MAGENTA+"Case 2 - classi known == RECIPROCHE"+Fore.RESET)
        if len(idx_reciprocal_class) == len(idx_known_class): # caso numero known = reciproci
            if max(idx_known_class) < min(idx_reciprocal_class):
                print(f"Case 2.1 - known index BEFORE reciprocal indices {max(idx_known_class)} <{ min(idx_reciprocal_class)}")
                # vuol dire che i reciproci sono posizionati dopo a seguire dei known 
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])
                for j in range(len(idx_reciprocal_class)):
                    color.append(color_reciprocal_class[j])
            else:
                print(f"Case 2.2 - known indices alternated with reciprocal indices {idx_known_class} {idx_reciprocal_class}")
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])
                    color.append(color_reciprocal_class[i])

        elif len(idx_reciprocal_class) > len(idx_known_class):  # caso numero knwon < reciproci = reci + all
            print(Fore.MAGENTA+"Case 3 - classes known < RECIPROCS = recip_i + recip_all"+Fore.RESET)
            if max(idx_known_class) < min(idx_reciprocal_class):
                # vuol dire che i reciproci sono posizionati dopo a seguire dei known 
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])
                for j in range(len(idx_reciprocal_class)-1):
                    color.append(color_reciprocal_class[j])

                color.append(color_reciprocal_all[0]) # aggiungi il colore del reciproco_all 
        
        elif len(idx_reciprocal_class) < len(idx_known_class): # coso known > reciprocal
            print(Fore.MAGENTA+f"Caso 4 - classes known > RECIPROCHE {len(idx_reciprocal_class)}"+Fore.RESET)
            if len(idx_reciprocal_class) == 1:
                 
                print(Fore.MAGENTA+"Caso 4.1 - classes RECIPRROC = 1 -> recip_all"+Fore.RESET)
                # caso in cui il reciproco è solo della classe all 
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])

                color.append(color_reciprocal_all[0])

            elif len(idx_reciprocal_class) > 1:
                print(Fore.MAGENTA+"Caso 4.2 - classes known == 2 -> brother of some class"+Fore.RESET)
                for i in range(len(idx_known_class)):
                    color.append(color_known[i])
                for i in range(len(idx_reciprocal_class)):
                    color.append(color_reciprocal_class[i])



    elif demo==True:
        for i in range(len(idx_known_class)):
            color.append(color_known[i])

    
    # verifica se è presente la classe unknown, quindi si è in open_set_testing , aggiungi il colore degli unknwon
    if bool_openset_testing and demo == False: 
        if len(idx_known_class) + len(idx_reciprocal_class) < len(map_label_to_class):
            print("Presente classe UNKNWON - aggiungi il colore ")
            color.append(color_unknown[0])
    
    elif bool_openset_testing:
        color.append(color_unknown[0])


    print(f"Array di colori selezionati - lunghezza {len(color)}:  {color}")



    if idx_reciprocal_class is not None and bool_openset_testing:
        unknown_class = idx_unknown
        print(f"Correct column Target class - etichetta reale per classe unknown = {unknown_class}")
        df['Correct_Target'] = np.where( (df['Target class'] == unknown_class) & (df['Predicted label'].isin(idx_reciprocal_class)),True, False )

        count = (df['Predicted label'] == unknown_class).sum()
        print(f"Numeri di elemeti con predicted unknown: {count}")
        
        count_target = (df['Correct_Target']).sum()
        print(f"Numeri di elemeti con Correct_Target: {count_target}")
    
    print(f"Lunghezza di colori {len(color)} == Lunghezza delle chaivi {len(map_label_to_class)}")
    
    colors_MAP = {}
    init = 0
    i = 0
    bar = []
    dist = []
    fraz = 1.0/(len(map_label_to_class.keys())-1)
    labels = list(map_label_to_class.keys())
    #for l, c in  map_label_to_class.items():
    for i, label in enumerate(labels):
        colors_MAP[int(label)] = color[i]
        bar.append([init,color[i]])
        dist.append(init)
        init = init + fraz
        i += 1
    
    dist[-1] = 1.0 
    bar[-1] = [1.0, color[-1]]

    print("Color Map \n",colors_MAP)
    print("Bar \n",bar)
    print("Dist \n",dist)


    count1 = (df['Predicted label'] == df['Target class']).sum()
    print(f"Numeri di elemeti con (predicted = real): {count1}")

    
    if df.get('Correct_Target') is  None:
        df["Symbol"] = df.apply( lambda row: "circle" if row["Target class"] == row["Predicted label"]  else "diamond", axis = 1) # passare riga per riga (axis=1)

    if df.get('Correct_Target') is not None:
        df["Symbol"] = df.apply( lambda row: "circle" if row["Target class"] == row["Predicted label"] or row["Correct_Target"] else "diamond", axis = 1) # passare riga per riga (axis=1)

        countcorrect = (df['Correct_Target']).sum()
        print("colonna Correct_Target è presente")
        print(f"Numeri di elemeti con Correct_Target: {countcorrect}")
        #df.loc[df['Correct_Target8'] == True, 'Symbol'] = "circle"
    

    countcicle = (df['Symbol'] == "circle").sum()
    print(f"Numeri di elemeti con circle in Symbol {countcicle}")
        
        #print(df[20:40])
    
    df["hover_text"] = df.apply(lambda row: f"Target class: {row['Target class']}<br>Predicted label: {row['Predicted label']}", axis=1)  # striga da stampare nell'etichetta tooltip
    
    # Creazione della colonna per le etichette  della legenda
    df["Legend Symbol"] = df["Symbol"].map({"circle": "Hit Classification", "diamond": "Miss Classification"})



    fig = go.Figure()

    # Aggiungiamo la traccia per i punti con "Hit Classification" (cerchio)
    fig.add_trace(go.Scatter(
        x=df[df["Symbol"] == "circle"]["x"],
        y=df[df["Symbol"] == "circle"]["y"],
        mode="markers",
        marker=dict(
            symbol="circle", size=8, # predette correttamente = cerchio
            #color=[colors_MAP[c] for c in  df[df["Symbol"] == "circle"]["Target class"]],  # colore basato sulla etichetta target
            color= df[df["Symbol"] == "circle"]["Target class"].astype(int),  # colore basato sulla etichetta target
            cmin= 0,
            cmax=len(map_label_to_class)-1,
            #color = color_list_circle,
            #colorscale='Viridis',
            colorscale = bar,
            opacity=0.5,
            #line=dict(width=1, colorscale="Viridis"),
            colorbar=dict(
                title="Target Classes",
                #tickvals=dist, # impostare i tick corrisponenti alle classi
                tickvals = list(range(len(map_label_to_class))),
                ticktext=[f"{key} : {map_label_to_class[key]}" for key in map_label_to_class.keys()], #etichettare i tick con i nomi delle classi
                len=1.0  # facoltativo: altezza della barra
            ),
            showscale=True,  # Mostra la colorbar
           

        ),
        name="Hit Classification",  # Nome nella legenda
        hovertext=df[df["Symbol"] == "circle"]["hover_text"],
        hoverinfo="text"
    ))

    # Aggiungiamo la traccia per i punti con "Miss Classification" (diamond)
    fig.add_trace(go.Scatter(
        x=df[df["Symbol"] == "diamond"]["x"],
        y=df[df["Symbol"] == "diamond"]["y"],
        mode="markers",
        marker=dict(
            symbol="diamond", size=8, 
            #color=df[df["Symbol"] == "diamond"]["Target class"], # colore basato sulla etichetta target
            #colorscale='Viridis',
            
            #color=[colors_MAP[c] for c in  df[df["Symbol"] == "diamond"]["Target class"] ],
            color= df[df["Symbol"] == "diamond"]["Target class"].astype(int),
            cmin= 0,
            cmax=len(map_label_to_class)-1,
            colorscale = bar,
            #color=color_list_diamond,
            opacity=0.7,
            line=dict(width=3, color="red"), 
            showscale=False,  # Disabilita la colorbar,
           
        ),
        name="Miss Classification",  # Nome nella legenda
        hovertext=df[df["Symbol"] == "diamond"]["hover_text"],
        hoverinfo="text"
    ))

    # Personalizzazione della legenda e layout
    
    fig.update_layout(
        title=f"t-SNE 2D -Predicted Classes \n {suffix}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y"
        ),
        legend=dict(
            x=0,  # [0-1] Posizione orizzontale: sinistra -> destra
            y=1,  # [0-1] Posizione verticale: basso -> alto
            xanchor="left",  # "left", "center", "right"
            yanchor="top",  # "top", "middle", "bottom"
        ),
        hovermode="closest"
    )
    if demo:
        path_dest = os.path.join(path_exp,f"Demo_plotly_tsn_2D_prob_{suffix}.html")
    else:
        path_dest = os.path.join(path_exp,f"plotly_tsn_2D_prob_{suffix}.html")
    
    fig.write_html(path_dest)


def run_visualizing( model,name_architecture, dataset_creator_obj, batch_size, phase , type_dataset ,num_classes, experiment , path_model, path_dest, path_file_config, map_label_to_class, idx_known_class, list_min=None, idx_reciprocal_class=None,idx_unknown=None, outlier=None, suffix="", classes_index_reciprocal=None, replace_idx_unknown=None, observ=None, list_distance = None, bool_openset_testing=False, demo=False):
    dataset_loader = None
    mode = None

    print(f"Map label to class: {map_label_to_class}, idx_known_class {idx_known_class}, idx_reciprocal_class {idx_reciprocal_class}, batch size {batch_size}")

    if phase == "visualization_test":
        if type_dataset == "known":
            dataset_creator_obj.create_DataLoader("test_known", batch_size)
            test = dataset_creator_obj.get_datasetLoader("test_known")
    
    if phase == "visualization_training":
        if type_dataset == "known":
            dataset_creator_obj.create_DataLoader("train_known", batch_size)
            test = dataset_creator_obj.get_datasetLoader("train_known")


    if phase == 'visualization':
        if type_dataset == "known":
            dataset_creator_obj.create_DataLoader("test_known", batch_size)
            test = dataset_creator_obj.get_datasetLoader("test_known")
        
        elif type_dataset == 'known_reciprocal':
            dataset_creator_obj.create_DataLoader("test_known_reciprocal", batch_size)
            test = dataset_creator_obj.get_datasetLoader("test_known_reciprocal")
        
        elif type_dataset == 'test_known_reciprocal_unknown':
            dataset_creator_obj.create_DataLoader("test_known_reciprocal_unknown", batch_size)
            test = dataset_creator_obj.get_datasetLoader("test_known_reciprocal_unknown")

        print(f" DETERMINISTIC TEST ON TEST LOADER in [visualization]")
        for img, label in test:
            print(label)
            break

        
        for img, label in test:
            print(label)
            break

    elif phase == "training":

        if type_dataset == "known_reciprocal":
            dataset_creator_obj.create_DataLoader("train_known_reciprocal", batch_size)
            test = dataset_creator_obj.get_datasetLoader("train_known_reciprocal")



    if test is None:
        print("Error! DatasetLoader is not present")
        sys.exit()
    print("DatasetLoader is present")

    print(f"DETERMINISTIC TEST on TEST LOADER in run_visualization")

    for img, label in test:
        print(type(img)) 
        print(Fore.GREEN+f"{label}"+Fore.RESET)
        break
    for img, label in test:
        print(Fore.GREEN+f"{label}"+Fore.RESET)
        break

    dataset_loader = test
    mode = "test"
        

    name_network = model.__class__.__name__
    print(f"Name network: {name_network}")

    vector_distrib_prob, target_array, predicted_label_array, features_array = extractor_features( name_architecture, model, dataset_loader, path_model, list_min, classes_index_reciprocal, replace_idx_unknown, observ, list_distance) 
    
    print(f"[extractor_features] [in Demo: {demo}] NUmero di elementi - lungheeza array etichette {len(target_array)}")
    corretti = sum(t == p for t, p in zip(target_array, predicted_label_array))
    print(f"[extractor_features] [in Demo: {demo}] Numero di elemnti corretti {corretti}")

    if demo:
        target_array_copy = target_array.copy()
        predicted_label_array_copy = predicted_label_array.copy()
        viewer_3D( vector_distrib_prob, target_array_copy, predicted_label_array_copy, map_label_to_class, path_dest,observ, idx_known_class, idx_reciprocal_class,idx_unknown, outlier, suffix, bool_openset_testing, demo)
        viewer_2D( vector_distrib_prob, target_array_copy, predicted_label_array_copy, map_label_to_class, path_dest,observ, idx_known_class, idx_reciprocal_class,idx_unknown, outlier, suffix, bool_openset_testing, demo)
    else:
        viewer_3D( vector_distrib_prob, target_array, predicted_label_array, map_label_to_class, path_dest,observ, idx_known_class, idx_reciprocal_class,idx_unknown, outlier, suffix, bool_openset_testing, demo)
        viewer_2D( vector_distrib_prob, target_array, predicted_label_array, map_label_to_class, path_dest,observ, idx_known_class, idx_reciprocal_class,idx_unknown, outlier, suffix, bool_openset_testing, demo)





    

def run_visualizing_known_with_pattern( model,name_architecture, dataset_creator_obj, batch_size , type_task  , path_model, path_dest, map_label_to_class, observ=""):
    if type_task == "plot_known_with_pattern":
        dataset_creator_obj.create_DataLoader("test_known_unknown", batch_size)
        test = dataset_creator_obj.get_datasetLoader("test_known_unknown")
        dataset_loader = test
    
        name_network = model.__class__.__name__
        print(f"Name network: {name_network}")

        features_array, target_array, predicted_label_array = extractor_features( name_architecture, model, dataset_loader, path_model) 
        viewer_3D( features_array, target_array, predicted_label_array, map_label_to_class, path_dest,observ )





    



