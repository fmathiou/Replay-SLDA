import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class SLDA(nn.Module):
    """SLDA model

    Attributes:
        model (Module): 
        W (Tensor): Weight matrix of classification layer.
        b (Tensor): Bias vector of classification layer.
        seen_classes (list): Labels of each encountered class.
        mean_features (list): Contains a mean feature vector for each encountered class.
        classes_counts (list): Contains the count number for each encountered class.
        precision_matrix (Tensor): Inverse of covariance matrix.

    """

    def __init__(self, base_model, output_classes=10):
        """Model constructor.

        Args:
            base_model (Module): Model to be used as feature extractor.
            output_classes (int, optional): . Defaults to 10.
        """
        
        super(SLDA, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fc_input = base_model.fc.in_features
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model = self.model.to(self.device)
        self.W = torch.tensor(np.zeros((output_classes, fc_input))).float()
        self.W = self.W.to(self.device)
        self.b = torch.tensor(np.zeros(output_classes)).float()
        self.b = self.b.to(self.device)
        self.seen_classes = []
        self.mean_features =[]
        self.classes_counts=[]
        
        # Calculate the inverse of the covariance matrix (precision matrix).
        # Covariance matrix is considered fixed and set to ones.
        cov_matrix = torch.tensor(np.ones((fc_input, fc_input))).float() 
        cov_matrix = cov_matrix.to(self.device)   
        epsilon = 1e-4
        diag_elements = torch.ones(cov_matrix.shape[1])
        I = torch.diag(diag_elements)
        I = I.to(self.device)
        reg_cov_matrix = (1 - epsilon)*cov_matrix + epsilon*I
        self.precision_matrix = torch.inverse(reg_cov_matrix)
        
    def forward(self, x):
        """Calculate ouptut."""
        x = self.model(x)       
        x = torch.squeeze(x)
        x = torch.transpose(x, 0, 1)
        x = torch.matmul(self.W, x) + torch.unsqueeze(self.b,-1)
        x = torch.transpose(x, 0, 1)
        return x
                
    def update_final_layer(self, x, y):
        """Update the weights of the classification layer.

        Args:
            x (Tensor): Batch of samples.
            y (Tensor): Corresponding labels.
        """
        with torch.no_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            unique_labels = torch.unique(y)
            for label in unique_labels: 
                instances_of_label = x[y==label]
                nr_of_instances = len(instances_of_label)
                features = self.model(instances_of_label)
                features_sum = torch.squeeze(torch.sum(features,0))
                
                # Update mean feature vectors
                if(label not in self.seen_classes): 
                    self.seen_classes += [label]
                    self.classes_counts += [nr_of_instances]
                    self.mean_features += [features_sum/nr_of_instances]
                else:
                    index = self.seen_classes.index(label)
                    mean_feature = self.mean_features[index]
                    ck = self.classes_counts[index]
                    self.classes_counts[index] += nr_of_instances
                    self.mean_features[index] = (ck * mean_feature + features_sum)/(ck+ nr_of_instances)

                # Update classification layer
                index = self.seen_classes.index(label)
                mean_feature = self.mean_features[index]
                self.W[label] = torch.matmul(self.precision_matrix, mean_feature)
                self.b[label] = -1/2 * torch.matmul(mean_feature, torch.matmul(self.precision_matrix, mean_feature))

def resNet18_model(output_classes, pretrained=True, freeze=True, bn_eval=True):
    """Create a ResNet-18 architecture model.

    Args:
        output_classes (int): Number of nodes in the final layer.
        pretrained (bool, optional): If true uses ImageNet weights. Defaults to True.
        freeze (bool, optional): If true freezes networks weights (besides classification layer). Defaults to True.
        bn_eval (bool, optional): If true sets batch normalization layers to evaluation mode. Defaults to True.

    Returns:
        ResNet: ResNet model.
    """
    model = models.resnet18(pretrained=pretrained)     
    if (freeze == True):
        for param in model.parameters():
            param.requires_grad = False
    fc_input = model.fc.in_features
    model.fc = nn.Linear(fc_input, output_classes)

    def set_bn_eval(module):
        """Set batch normalization layers to evaluation mode."""
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    if (bn_eval==True):
        model.apply(set_bn_eval)
    return model