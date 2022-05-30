import random
from torch.utils.data import Sampler

class CL_Sampler(Sampler[int]):
    """Adjusts data loading sequence to mimic a non-i.i.d. stream of data.

    It specifies the sequence of indicies used for data loading such as 
    to correspond to an ordered dataset where instances are grouped by class.
    This is needed in online continual learning where the assumption 
    of a non-i.i.d. stream of data is made.

    Attributes:
        indices (list): Utilized for sampling instances.
        targets (list): Dataset's target labels.
    """
    
    def __init__(self, dataset):
        self.indices=list([])
        self.targets = dataset.targets
        self.lenght = len(dataset)

    def __iter__(self):
        unique_labels = list(set(self.targets))
        random.shuffle(unique_labels)
        # get indices for ordering
        for label in unique_labels:
            # get indices of class i
            class_indices = list(filter(lambda x: self.targets[x] == label, range(len(self.targets))))
            # shuffle class indices
            random.shuffle(class_indices)
            # add class indices to list of total indices 
            self.indices += class_indices
        return iter(self.indices)
    
    def __len__(self):
        return self.lenght

