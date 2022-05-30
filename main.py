from torchvision import transforms, datasets
import torch.optim as optim
from models import resNet18_model
from training import replay_slda_v2
from training import evaluate


def main():
    data_dir = './data'
    # Rest of transofrmations is performed inside the training algorithm
    train_transform = transforms.ToTensor() 
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224), 
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    trainset = datasets.CIFAR100(root=data_dir, train=True, 
                                 transform=train_transform)
    testset = datasets.CIFAR100(root=data_dir, train=False, 
                                transform=test_transform)

    # Train the model
    output_classes = 100
    model = resNet18_model(output_classes, freeze=False)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    sl_model = replay_slda_v2(model, trainset, optimizer, batch_size=10, 
                              batch_loops=1, max_samples=600,
                              replay_size=10, output_classes=100)
    evaluate(model, testset)
    evaluate(sl_model, testset)

if __name__ == '__main__':
    main()