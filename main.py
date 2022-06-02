from torchvision import transforms, datasets
import torch.optim as optim
from models import resNet18_model
from training import *


def main():
    data_dir = './data'
    trainset = datasets.CIFAR100(root=data_dir, train=True, 
                                 transform=transforms.ToTensor())
    output_classes = len(set(trainset.targets))
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224), 
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    testset = datasets.CIFAR100(root=data_dir, train=False, 
                                transform=test_transform)
    # Training transformations are used within the training algorithm
    # to avoid storing upscaled instances
    train_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])]) 
    # Train the model
    model = resNet18_model(output_classes, freeze=False)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    sl_model = replay_slda_v2(model, trainset, optimizer, batch_size=10, 
                              batch_loops=1, max_samples=600,
                              replay_size=10, output_classes=output_classes,
                              transform=train_transform)

    accuracy = evaluate(sl_model, testset)
    print(f'Accuracy of the model on the test set: {accuracy}%')

if __name__ == '__main__':
    main()