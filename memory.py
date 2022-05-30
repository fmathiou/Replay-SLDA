from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random


class Memory(Dataset):
    """Memory buffer used for rehearsal.

    Attributes:
        max_samples (int): Maximum allowed number of samples to be stored.
        reservoir (list): Contains stored instances in the form [instance, label].
        seen_samples (int): Number of instances ecnountered.
        transform (Tansform): Transformation operation when replaying.
    """
    
    def __init__(self, max_samples=200):
        super(Memory, self).__init__()
        self.max_samples = max_samples
        self.reservoir =[]
        self.seen_samples = 0
        self.transform = transforms.Compose([transforms.Resize(256), 
                                             transforms.CenterCrop(224),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def __len__ (self):
        return len(self.reservoir)
        
    def __getitem__(self, index):
        sample = self.reservoir[index]
        return sample
        
    def reservoir_sampling(self, samples, labels):
        """Perform reservoir sampling."""
        nr_of_smaples = labels.shape[0]
        for i in range(nr_of_smaples):
            if(self.seen_samples < self.max_samples):
                self.seen_samples += 1
                self.reservoir.append([samples[i], labels[i]])
            else:
                self.seen_samples += 1   
                random_index = random.randrange(self.seen_samples)
                if(random_index < self.max_samples):
                    self.reservoir[random_index] = [samples[i], labels[i]]

    def random_replay(self, batch_size):
        """Draw instances from the reservoir uniformly at random.

        Args:
            batch_size (int): Numbber of instances to sample.

        Returns:
            tuple: Samples and corresponding labels.
        """
        if(len(self.reservoir)) >= batch_size:
            random_indices = random.sample(range(len(self.reservoir)), batch_size)
            batch = list(map(self.__getitem__, random_indices))
            samples = list(map(lambda x: x[0], batch))
            samples = torch.stack(samples)
            samples = self.transform(samples)
            labels = list(map(lambda x: x[1], batch)) 
            labels = torch.stack(labels)
            
        else:
            samples = torch.tensor([])
            labels = torch.tensor([])
            
        return samples, labels