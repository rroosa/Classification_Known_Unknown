import torch 
from torch.utils.data import Dataset, ConcatDataset 
from collections import Counter
import torch
from torch.utils.data import Dataset, ConcatDataset
from colorama import Fore
class ShuffledConcatDataset(Dataset):
    def __init__(self, datasets,targets, seed=42):
        self.concat = ConcatDataset(datasets)  # concatenated in original sequence
        self.datasets = self.concat.datasets  # Direct access to sub-datasets.
        
        self.seed = seed
        
        g = torch.Generator()
        g.manual_seed(seed)
        self.shuffled_indices = torch.randperm(len(self.concat), generator=g).tolist()
        self.targets = targets[self.shuffled_indices] #Permute the targets in the same way as well.
        self.cumulative_sizes = self.concat.cumulative_sizes
        print(f"Count by classes of the concatenated datasets {Counter(self.targets)}")
    

    
    def __getitem__(self, idx):
        shuffled_idx = self.shuffled_indices[idx]
        dataset_idx, sample_idx = self._get_dataset_index(shuffled_idx)
        sample = self.datasets[dataset_idx][sample_idx]
        return sample
    
    def _get_dataset_index(self, idx):
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                prev_size = 0 if i == 0 else self.cumulative_sizes[i - 1]
                return i, idx - prev_size
        raise IndexError("Index out of range")

    def __len__(self):
        return len(self.concat)
    
    def get_concat_targets(self):
        return self.targets 

    def get_sample_info(self, idx):

        try:
            if idx >= len(self.shuffled_indices):
                raise IndexError(Fore.RED+f"Index out of range: {idx}"+Fore.RESET)
        except IndexError as e:
            print("Error:", e)
            return (None,None,None)

        # Find the original index before shuffling
        original_idx = self.shuffled_indices[idx]

        # retrieve the correct sub-dataset and its internal index.
        dataset_idx, sample_idx = self._get_dataset_index(original_idx)

        dataset = self.datasets[dataset_idx]

        #Check that the dataset has the method get_sample_info before calling it.
        if not hasattr(dataset, "get_sample_info"):
            raise AttributeError(f"The dataset {dataset_idx} does not have the method get_sample_info")

        # Call and return the information.
        return dataset.get_sample_info(sample_idx)

