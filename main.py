from torchvision import transforms, datasets
import torch.optim as optim
from models import resNet18_model
from training import *
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import json


# Override init function of Subset class 
# to include attribute targets. Needed for
# the crh dataset
class custom_subset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        targets = dataset.targets
        self.targets = [targets[i] for i in indices]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_alg',
        type=str,
        choices = ['rehearsal', 'slda', 'replay-slda'],
        default='slda',
        help='training algorithm'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices = ['cifar10', 'cifar100', 'crh'],
        default='cifar10',
        help='image dataset'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='learning rate for SGD'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='number of samples in the batch'
    )
    parser.add_argument(
        '--batch_loops',
        type=int,
        default=1,
        help='number of times to iterate on a given batch'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=600,
        help='number of samples stored for replay'
    )
    parser.add_argument(
        '--replay_size',
        type=int,
        default=10,
        help='size of batch replayed from memory'
    )
    parser.add_argument(
        '--print_interval',
        type=float,
        default=1/4,
        help='print training info at specified intervals of \
              the dataset size'
    )
    parser.add_argument(
        '--save_model',
        default=False,
        action='store_true',
        help='save trained model'
    )

    opt = parser.parse_args()
    data_dir = './data' 
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224), 
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    # Training transformations are used within the training algorithm
    # to avoid storing upscaled instances
    train_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])                                                            
         
    if opt.dataset == 'cifar10':
        training_set = datasets.CIFAR10(root=data_dir, train=True, 
                                        transform=transforms.ToTensor(), download=True)
        testset = datasets.CIFAR10(root=data_dir, train=False, 
                                   transform=test_transform)
    elif opt.dataset == 'cifar100':
        training_set = datasets.CIFAR100(root=data_dir, train=True, 
                                         transform=transforms.ToTensor())
        testset = datasets.CIFAR100(root=data_dir, train=False, 
                                    transform=test_transform, download=True)
    elif opt.dataset == 'crh':
        training_set = datasets.ImageFolder(root='./data/histology', transform = transforms.ToTensor())
        testset = datasets.ImageFolder(root='./data/histology', transform = test_transform)
        train_indices, test_indices = train_test_split(np.array(range(len(training_set))), 
                                                       test_size=0.2, train_size=0.8, 
                                                       random_state=42, shuffle=True) 
        training_set = custom_subset(training_set, train_indices)
        testset = custom_subset(testset, test_indices)

    print(f'{opt.dataset} stream selected')
    output_classes = len(set(training_set.targets))
    model = resNet18_model(output_classes)
    train_param = {'batch_size': opt.batch_size, 'batch_loops': opt.batch_loops, 'max_samples': opt.max_samples, 
                   'replay_size': opt.replay_size}
    # Train the model
    if opt.train_alg == 'rehearsal':
        print('Training parameters:\n', train_param)

        optimizer = optim.SGD(model.parameters(), lr=opt.lr)
        trained_model = rehearsal_train(model, training_set, optimizer, batch_size=opt.batch_size, batch_loops=opt.batch_loops, 
                                        max_samples=opt.max_samples, replay_size=opt.replay_size, transform=train_transform, 
                                        criterion = None, print_interval = opt.print_interval)
    elif opt.train_alg == 'slda':
        model = SLDA(model)
        trained_model = slda_train(model, training_set, train_transform, batch_size=opt.batch_size, 
                                   print_interval=opt.print_interval)
    else :
        print('Training parameters:\n', train_param)
        optimizer = optim.SGD(model.parameters(), lr=opt.lr)
        trained_model = replay_slda_v2(model, training_set, optimizer, batch_size=opt.batch_size, batch_loops=opt.batch_loops,
                                       max_samples=opt.max_samples, replay_size=opt.replay_size, transform=train_transform,
                                       criterion = None, print_interval=opt.print_interval)

    accuracy = evaluate(trained_model, testset)
    print(f'Accuracy of the model on the test set: {accuracy}%')

    if (opt.save_model == True):
        print('Saving model to "./saved_models"')
        PATH = './saved_models'
        current_time = datetime.now()
        current_time = current_time.strftime("%d-%m-%Y %H.%M")
        if not os.path.exists(PATH):
              os.makedirs(PATH)
        torch.save(model.state_dict(), PATH + f'/{opt.dataset}_{opt.train_alg}_{current_time}.pth')
        # Save json file with the training parameters
        if (opt.train_alg != 'slda'):
            with open(PATH + f'/{current_time}.json', 'w') as fp:
                json.dump(train_param, fp)

if __name__ == '__main__':
    main()