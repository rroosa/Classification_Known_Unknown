from torchvision import datasets, transforms
from sklearn.datasets import fetch_lfw_people
import sys
def get_dataset_MNIST(folder, train = True, transform= None):

	return datasets.MNIST(
            root =f"{folder}",
            train = train,
            download = True,
            transform = transform) 

        
        

collection_datasets = { "MNIST": get_dataset_MNIST    
						}


