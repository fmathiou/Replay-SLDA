import torch
import torch.nn as nn
import time
from sampler import CL_Sampler
from torch.utils.data import DataLoader
from torchvision import transforms
from memory import Memory
from models import SLDA

def rehearsal_train(model, trainset, optimizer, batch_size, batch_loops, 
                    max_samples, replay_size, replay=True, criterion = None):
    """Train the 'model' sequentially using rehearsal.

    The 'model' is trained through rehearsal on the 'trainset' which is ordered
    by class to represent a non-i.i.d. stream of data. If 'replay' is set to
    False, this corresponds to naive continual traning on the stream.

    Args:
        model (torchvision.models): Model to be trained.
        trainset (Dataset): Training set.
        optimizer (torch.optim): Optimization algorithm.
        batch_size (int): Number of samples in the batch.
        batch_loops (int): Number of optimization steps on a given batch.
        replay (bool): If true activates rehearsal. Defaults to True.
        max_samples (int): Maximum number of samples stored for rehearsal.
        replay_size (int): Number of samples replayed from memory.
        criterion ( Loss, optional): Loss function. Defaults to None.
    """

    start = time.time()
    if criterion == None:
        criterion = nn.CrossEntropyLoss()
    sampler = CL_Sampler(trainset)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    total_batches = len(train_loader)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if (replay == True): memory = Memory(max_samples)
        
    # train on stream
    for i, data in enumerate(train_loader, 0):
        inputs, labels =  data[0].to(device), data[1].to(device)
        transformed_inputs = transform(inputs.clone())
        for j in range(batch_loops):
            if (replay == True):
                replayed_features, replayed_labels = memory.random_replay(replay_size)

                # if replay batch not empty augment main batch
                if replayed_features.shape[0]!=0:
                    augmented_inputs = torch.vstack((transformed_inputs, replayed_features))
                    augmeneted_labels = torch.hstack((labels, replayed_labels))

            optimizer.zero_grad()
            outputs = model(augmented_inputs)
            loss = criterion(outputs, augmeneted_labels)
            loss.backward()
            optimizer.step()
       
        if (replay == True):
            memory.reservoir_sampling(inputs, labels)
            
        # Print statistics
        n = total_batches//4
        if i % n == n-1:
            print('Mini-batch:', f'{i + 1}/{total_batches}')

    finish = time.time()
    print('Finished training')
    elapsed_time = finish-start
    print(f'Training time: {elapsed_time/60:.2f} minutes')

def slda_train(model, trainset, batch_size):
    """Perform training with Deep Streaming Linear Discriminant Analysis.

    Args:
        model (torchvision.models): Model to be trained.
        trainset (Dataset): Training set.
        batch_size (int): Number of samples in the batch.
    """
    start = time.time()
    sampler = CL_Sampler(trainset)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    total_batches = len(train_loader)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # train on stream
    for i, data in enumerate(train_loader, 0):
        inputs, labels =  data[0].to(device), data[1].to(device)
        transformed_inputs = transform(inputs.clone())
        model.update_final_layer(transformed_inputs, labels)
        
       # Print statistics
        n = total_batches//4
        if i % n == n-1:
            print('Mini-batch:', f'{i + 1}/{total_batches}')
    
    finish = time.time()
    print('Finished training')
    elapsed_time = finish-start
    print(f'Training time: {elapsed_time/60:.2f} minutes')

def replay_slda_v2(model, trainset, optimizer, output_classes, batch_size, 
                   batch_loops, max_samples, replay_size, criterion = None):
    """Continual training using Replay-SLDA (v2) algorithm.

    Args:
        model (torchvision.models): Model to be trained.
        trainset (Dataset): Training set.
        optimizer (torch.optim): Optimization algorithm.
        output_classes (int): Number of ouptut classes.
        batch_size (int): Number of samples in the batch.
        batch_loops (int): Number of optimization steps on a given batch.
        max_samples (int): Maximum number of samples stored for rehearsal.
        replay_size (int): Number of samples replayed from memory.
        criterion ( torch.nn.modules.loss, optional): Loss function. Defaults to None.

    Returns:
        SLDA: Final model.
    """
    start = time.time()
    if criterion == None:
        criterion = nn.CrossEntropyLoss()
    sampler = CL_Sampler(trainset)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    total_batches = len(train_loader)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    memory = Memory(max_samples)

    # train on stream
    print('Training network on the stream through rehearsal')
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels =  data[0].to(device), data[1].to(device)
        transformed_inputs = transform(inputs.clone())
        
        for j in range(batch_loops):
            replayed_features, replayed_labels = memory.random_replay(replay_size)

            # if replay batch not empty augment main batch
            if replayed_features.shape[0]!=0:
                augmented_inputs = torch.vstack((transformed_inputs, replayed_features))
                augmented_labels = torch.hstack((labels, replayed_labels))
            else:
                augmented_inputs = transformed_inputs.clone()
                augmented_labels = labels.clone()

            optimizer.zero_grad()
            outputs = model(augmented_inputs)
            loss = criterion(outputs, augmented_labels)
            loss.backward()
            optimizer.step()           
        memory.reservoir_sampling(inputs, labels)
        
        # Print statistics
        n = total_batches//4
        if i % n == n-1:
            print('Mini-batch:', f'{i + 1}/{total_batches}')
     
    with torch.no_grad():
        print('Training classification layer on the stored instances through SLDA')
        # create data loader for buffer
        loader = DataLoader(memory, batch_size=10)
        # create slda model
        slda_model = SLDA(base_model=model,output_classes=output_classes)
        for i, data in enumerate(loader, 0):
            inputs, labels =  data[0].to(device), data[1].to(device)
            transformed_inputs = transform(inputs.clone())
            slda_model.update_final_layer(transformed_inputs, labels)
             
    finish = time.time()
    print('Finished training')
    elapsed_time = finish-start
    print(f'Training time: {elapsed_time/60:.2f} minutes')
    
    return slda_model

def evaluate(model, dataset, batch_size=32):
    """Calculate the accuracy the provided dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    data_loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for data in data_loader:
            test_images, test_labels = data[0].to(device), data[1].to(device)
            outputs = model(test_images)
            _, predicted_classes = torch.max(outputs, 1)
            total += test_labels.size(0)
            correct += (predicted_classes == test_labels).sum().item()
    accuracy = 100 * correct / total 
    print(f'Accuracy of the network on the test images:{accuracy}%')
    return accuracy