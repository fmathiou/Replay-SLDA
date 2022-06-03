import random
from torch.utils.data import Sampler

class CL_Sampler(Sampler[int]):
    """Adjusts data loading sequence to mimic a non-i.i.d. data stream.

    It specifies the sequence of indicies used for data loading such as 
    to correspond to an ordered dataset where instances are grouped by class.
    This is useful in online continual learning where the assumption 
    of a non-i.i.d. data stream is made.

    Attributes:
        indices (list): Indices used for data loading.
        targets (list): Labels of dataset's instances.
    """
    
    def __init__(self, dataset):
        self.indices=list([])
        self.targets = dataset.targets
        self.lenght = len(dataset)

    def __iter__(self):
        unique_classes = list(set(self.targets))
        # Shuffle to change class order between experiments
        random.shuffle(unique_classes)
        for class_ in unique_classes:
            class_indices = list(filter(lambda x: self.targets[x] == class_, range(len(self.targets))))
            # Shuffle to change within-class sample order between experiments
            random.shuffle(class_indices)
            self.indices += class_indices
        return iter(self.indices)
    
    def __len__(self):
        return self.lenght

