import argparse
import torch
import json
import requests
import os
from utility import *

import tarfile
from colorama import Fore
import configparser

####--------------------------------------------------------------------------------------###
## Script to individually download classes from the ImageNet dataset using the APIs ##
## starting from their class names.                                                  ##
## The images are downloaded as compressed folders and extracted inside the          ##
## Imagenet_photos directory. 
###--------------------------------------------------------------------------------------####
class Imagenet_Downloader():

    def __init__(self,path):
        # lest of classes in ImageNet
        self.url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

        #self.IMAGENET_API_WNID_TO_URLS = lambda wnid: f'http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}'

        self.API = lambda wnid: f'https://image-net.org/data/winter21_whole/{wnid}.tar'

        self.file_list_classes = self.url.split("/")[-1] # imagenet_class_index.json
        
        self.root = path
        self.folder_photos = None

        self.__download_list_classes()
        self.json_info_classes = self.__read_file_json()
        
   
    def get_folder_photos(self):
        return self.folder_photos
    
    def set_folder_photos(self, folder):
        self.folder_photos = folder
    


    def __download_list_classes(self):
        #  Check if the JSON file is not already present
        file_path = os.path.join(self.root, self.file_list_classes)

        if os.path.exists(file_path):
            print(f"File {self.file_list_classes} is present")
        else:
            file_json = requests.get(self.url).json()

            # save  file json
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(file_json, file, ensure_ascii=False, indent=4)
                print(f"File {self.file_list_classes} salves with success!")

    def __read_file_json(self):
        file_path = os.path.join(self.root, self.file_list_classes)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file) 
        except FileNotFoundError:
            print(f" Error: file {self.file_list_classes} isn't present")
        except json.JSONDecodeError:
            print(f"Error: file {self.file_list_classes} doesn't contain valid JSON")
        except Exception as e:
            print(f"Error {e}")

    def __find_wnid_for_classname(self, class_name):

        for key, value in self.json_info_classes.items(): #{"id": ["wnid", "name_classes"]}
            if class_name.lower() == value[1].lower():
                return value[0], value[1]

        return None, None


    def download_photos_for_name(self, name_class):
        #### ----------------------------------------------------------------------------------------##############
        # 1 - Search in the JSON file if the class is present,
        #     if present, retrieve the WNID
        # 2 - Then download the file containing the list of URLs for that class
        # 3 - Create the appropriate folder
        # 4 - Download each image and save it in the folder
        ###------------------------------------------------------------------------------------------##############
        print(Fore.CYAN+f" STEP 1/5 - Search class for name ..."+Fore.RESET)
        synset_id, class_name = self.__find_wnid_for_classname(name_class)

        if synset_id is not None:
            print(Fore.CYAN+f" STEP 2/5 - Create folder for class_name [{class_name}] ..."+Fore.RESET)
            path_folder = os.path.join(self.root, self.folder_photos)
            path_folder_dest = os.path.join(path_folder, class_name)
            create_Folder(path_folder)

            api = self.API(synset_id) # 
            output_file = f"{class_name}.tar"

            print(Fore.CYAN+f" STEP 3/5 - Request download file.tar for [{class_name}] ..."+Fore.RESET)
            response = requests.get(api, stream=True)
            if response.status_code == 200:
                with open(output_file, "wb") as file: # create file in binary mode 
                    for chunk in response.iter_content(chunk_size=1024): # Iterate in blocks of 1024 over the response stream.”
                        file.write(chunk) # write on file
                print(Fore.GREEN+f"Download completed: {output_file}"+Fore.RESET)
            else:
                print(Fore.RED+f"Error {response.status_code}: Unable to download tar file"+Fore.RESET)

            print(Fore.CYAN+f" STEP 4/5 - Extract from {output_file} in folder [{self.folder_photos}/{class_name}] ..."+Fore.RESET)

            with tarfile.open(output_file, "r") as tar:
                tar.extractall(path_folder_dest)
            print(Fore.GREEN+"Extraction completed"+Fore.RESET)

            print(Fore.CYAN+f" STEP 5/5 - Remove file tar [{output_file}] ..."+Fore.RESET)
            remove_file(output_file)    # delete file tar

        else:

            print(f"The class name [{name_class}] is not present")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Downloader')
    parser.add_argument('--class_name', type=str, required= True)
    parser.add_argument('--folder', type=str, choices=["Imagenet_photos", "Imagenet_unknown"], required= True, help="Name folder es. Imagenet_photos | Imagenet_unknown")

    args = parser.parse_args()

    class_name = args.class_name
    folder = args.folder

    config = configparser.ConfigParser()
    config.read('config.ini')

    path_home_dataset = config["absolute_path"]["datasets"]
    print(f"Absolute path for datasets {path_home_dataset}")

    downloader_obj = Imagenet_Downloader(path_home_dataset)

    downloader_obj.set_folder_photos(folder)
    folder_photos = downloader_obj.get_folder_photos()
    create_Folder(folder_photos)
    
    downloader_obj.download_photos_for_name(class_name)

