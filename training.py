from torch.utils.data import DataLoader
from meters import AverageValueMeter, Plotter_Meters
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import os
import torch
import time
#from EarlyStopping import EarlyStopping
import torch.nn.functional as F
import threading
import subprocess 
import signal
import torch.optim as optim
from loss_functions import *

def stop_tensorboard(port):
    # Find the PID of the TensorBoard process listening on the port
    try:
        result = subprocess.check_output(f"lsof -t -i:{port}", shell=True)
        pid = result.strip().split(b'\n')  #
        pids = [int(line) for line in pid]
        # end the TensorBoard process
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
            print(f" TensorBoard process on port {port} - killed.")
    except subprocess.CalledProcessError:
        print(f"No process found on the port {port}.")

def start_tensorboard(logdir_root, port):
    # Check if TensorBoard is already running and, if so, terminate the process
    stop_tensorboard(port)
    # Start the new TensorBoard process
    print(f"Log dir root {logdir_root}")
    command = f"tensorboard --logdir={logdir_root} --port={port} --host=localhost"
    print(f"Avvio TensorBoard con il comando: {command}")  # Debug
    subprocess.Popen(command, shell=True) 




def run_training(device,  model, optimizer, criterion, scheduler, num_epochs, batch_size, dataset_creator_obj, logdir_root, port_logs , type_dataset = None, experiment='experiment_[1]', path_folder=None, path_file_config=None, continue_training= False ):
    
    start_epoch = 0
    print(f"Start_epochs {start_epoch}, Num_epochs {num_epochs}, Bool continue training {continue_training}")

    tb_thread = threading.Thread( target = start_tensorboard, args = (logdir_root, port_logs) )
    tb_thread.start()

    
    print("tipo modelo:",type(model))
    model.to(device)                        
                 
                                             


    if continue_training:
        name_model = f"model_{experiment}.pth"
        path = os.path.join(path_folder,experiment)
        path_model = os.path.join(path, name_model)
        checkpoint = torch.load(path_model, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            start_epoch = epoch + 1

        if isinstance(checkpoint, dict) and "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if not isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        
        print(f"Continue - Start_epochs {start_epoch}, Num_epochs {num_epochs}")

    train_loader = None 
    validation_loader = None
    
    if type_dataset == 'known':
        #----- Get DATALOADER -----------------------------
        train_loader = dataset_creator_obj.get_datasetLoader("train_known")
        validation_loader = dataset_creator_obj.get_datasetLoader("validation_known")
    
    elif type_dataset == 'known_reciprocal':
        train_loader = dataset_creator_obj.get_datasetLoader("train_known_reciprocal")
        validation_loader = dataset_creator_obj.get_datasetLoader("validation_known_reciprocal")


   
    print(type(train_loader.dataset))  
    print(type(validation_loader.dataset)) 

    print("Start training ...")
    start = time.time()
    for images, labels in train_loader:
        print(f"Check shape of batch: {images.shape}") 
        break

    loader = {                              #dict with loader of training e test
        'train' : train_loader,
        'validation' : validation_loader
        }

    #------meters-----------------------------------
    plotter_obj = Plotter_Meters()
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    loss_epochs = 0

    #------logs-------------------------------------
    logdir = os.path.join(logdir_root, experiment )
    writer = SummaryWriter(logdir)
    os.makedirs(logdir, exist_ok=True)

    #-----device--------------------------------------
    #device = f"cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")



    #-------------------------------------------------------


    global_step = 0
    effective_epoch = 0
    
    for e in range(start_epoch, num_epochs):
        print(f"[Epoch - {e+1}/{num_epochs}]")
        for mode in ['train','validation']:
            print(f"   Mode -> {mode}")
            loss_meter.reset()
            acc_meter.reset()
            loss_epochs = 0

            model.train() if mode == 'train' else model.eval()

            with torch.set_grad_enabled(mode == 'train'): # active gradients only in  train mode

                for i, batch in enumerate(loader[mode]):

                    x = batch[0].to(device) #
                    y = batch[1].to(device) # label



                    if mode == 'train':
                        optimizer.zero_grad()


                    output = model(x)
                    
                    features = None
                    if isinstance(output, tuple):
                        features, output = output
                    else:
                        output = output
                        

                    n = x.shape[0]  # num sample in batch
                    global_step += n

                    if criterion.__class__.__name__ == "JSD_Loss":
                        #print(f"Function Loss -JDS_Loss-") 
                        predicted_prob = F.softmax(output, dim=1) 
                        # trasform label in one-hot
                        target_prob = F.one_hot(y, num_classes=predicted_prob.shape[1]).float()
                        loss = criterion(target_prob, predicted_prob)
                    else:
                        loss = criterion(output, y)



                    if mode == 'train':
                        
                        loss.backward()    # Calculation of partial derivatives with respect to the gradients
                        optimizer.step()   # update parameters




         
                    output = F.softmax(output, dim=1) # trasform output in probprobabilitis

                    acc = accuracy_score(y.to('cpu'), output.to('cpu').max(1)[1]) 
                    loss_meter.add(loss.item(),n)
                    acc_meter.add(acc,n)

                    loss_epochs += loss.item()


                    #log batch per batch only for training
                    if mode == 'train':
                        #-------------- TENSORBOAD--------------------------------------------------
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step = global_step)
                        writer.add_scalar('accuracy_epoch/train', acc_meter.value(), global_step = e)
                        writer.add_scalar('loss/train', loss_meter.value(), global_step = global_step)
                        writer.add_scalar('loss_epochs/train', loss_epochs/len(loader[mode]), global_step = e)
                        #---------------PLOTTER-------------------------------------------------------
                        plotter_obj.append_accuracy(mode, acc_meter.value(), global_step)
                        plotter_obj.append_loss(mode, loss_meter.value(), global_step)
                        plotter_obj.append_loss_epoch(mode, loss_epochs/len(loader[mode]), e )

            # end of iteration over all batches of that epoch, for train or validation
            if mode == 'train' and scheduler is not None and  not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    print("scheduler present - exec after each epoch")
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        print("scheduler present -ReduceLROnPlateau")
                        scheduler.step(loss_meter.value()) # update learning rate 
                    else:
                        scheduler.step()

            # logg
            #-------------- TENSORBOAD--------------------------------------------------
            writer.add_scalar(f'accuracy/{mode}', acc_meter.value(), global_step = global_step)
            writer.add_scalar(f'accuracy_epoch/{mode}', acc_meter.value(), global_step = e)
            writer.add_scalar(f'loss/{mode}', loss_meter.value(), global_step = global_step)
            writer.add_scalar(f'loss_epochs/{mode}', loss_epochs/len(loader[mode]), global_step = e)
            #---------------PLOTTER-------------------------------------------------------
            plotter_obj.append_accuracy(mode, acc_meter.value(), global_step)
            plotter_obj.append_loss(mode, loss_meter.value(), global_step)
            plotter_obj.append_loss_epoch(mode, loss_epochs/len(loader[mode]), e )




        effective_epoch += 1
        end = time.time() 
        tot_time = end - start
        print(f"Training completed - total time : {tot_time} sec")
        print(f"effective_epoch:{effective_epoch}/{num_epochs}")
        plotter_obj.set_timeTraining(tot_time)
        plotter_obj.set_effective_epochs(effective_epoch,num_epochs)
    
        # salvare modello 
        path = os.path.join(path_folder,experiment)
        
        name_model = f"model_{experiment}_epoch_[{e}].pth"
        
        
        path_model = os.path.join(path, name_model)

        """    
        print(f"Save model - [{name_model}]")
        if scheduler is not None:
            torch.save({"epoch":num_epochs-1, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict(), "scheduler_state_dict":scheduler.state_dict()},path_model)
        else:
            torch.save({"epoch":num_epochs-1, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict()},path_model)
        """

       
        # save plots
        print("Save plots")
        plotter_obj.save_meters_plot_accuracy( path, experiment)
        plotter_obj.save_meters_plot_loss( path, experiment)
        plotter_obj.save_meters_plot_loss_epoch(path, experiment)

        # save information by  plotter_obj
        print("Save Results last loss/accuracy")
        plotter_obj.saveLastPerformance( key= experiment , path_config= path_file_config )




    # end of all epochs
    end = time.time() 
    tot_time = end - start
    print(f"Training completed - total time : {tot_time} sec")
    print(f"effective_epoch:{num_epochs}/{num_epochs}")
    plotter_obj.set_timeTraining(tot_time)
    plotter_obj.set_effective_epochs(num_epochs,num_epochs)
   
    # salvare modello 
    name_model = f"model_{experiment}.pth"
    print(f"Save model Final - [{name_model}]")
    path = os.path.join(path_folder,experiment)
    path_model = os.path.join(path, name_model)
    if scheduler is not None:
        torch.save({"epoch":num_epochs-1, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict(), "scheduler_state_dict":scheduler.state_dict()},path_model)
    else:
        torch.save({"epoch":num_epochs-1, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict()},path_model)


    # salvare i plot
    print("Save plots")
    plotter_obj.save_meters_plot_accuracy( path, experiment)
    plotter_obj.save_meters_plot_loss( path, experiment)
    plotter_obj.save_meters_plot_loss_epoch(path, experiment)

    # save information tramite oggetto plotter_obj
    if num_epochs != 0:
        print("Save Results last loss/accuracy")
        plotter_obj.saveLastPerformance( key= experiment , path_config= path_file_config )










