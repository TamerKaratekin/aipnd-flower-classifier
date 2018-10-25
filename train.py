import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, models, transforms
import numpy as np
import sys, time, os, copy, json
# from utils import (train_model, load_datasets, device_type)
# might want to seperate the code into these parts in future
from collections import OrderedDict

## sample call on command line: python train.py flowers --hu 4096 1024 --ckpt_name='checkpoint.pth' --gpu_mode=True --lr=0.0009 --epochs=4 --arch='vgg16'

def parse_args():
    parser = argparse.ArgumentParser(description="Trains a network on a dataset of images and saves the model to a checkpoint")
    parser.add_argument('data_dir', action='store', default='flowers', type=str, help='data directory of the training, validation, and test images')
    parser.add_argument("--save_dir", action="store", dest="save_dir", default="." , help = "Set directory to save checkpoints")
    parser.add_argument("--arch", action="store", dest="arch", default="vgg16" , choices=["vgg16", "resnet18"], help = "Set architechture('vgg16' or 'resnet18')")
    parser.add_argument("--lr", action="store", dest="learning_rate", type=float, default=0.001 , help = "Set learning rate, default at 0.001")
    parser.add_argument("--hu", action="store", dest="hidden_layers", nargs='+', type=int, default=None, help = "Set number of hidden units as integer array")
    parser.add_argument("--epochs", action="store", dest="epochs", type=int, default=3 , help = "Set number of epochs, default 3 ")
    parser.add_argument('--gpu_mode', default=False, type=bool, help='Set the gpu mode')
    parser.add_argument('--ckpt_name', dest="checkpoint_name", default='checkpoint.pth', type=str, help='Set the save name')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    checkpoint_name = args.checkpoint_name
    gpu_mode = args.gpu_mode
    
    print('='*10+'Params'+'='*10)
    print('Data dir:      {}'.format(data_dir))
    print('Arch:         {}'.format(arch))
    print('Hidden layers: {}'.format(hidden_layers))
    print('Learning rate: {}'.format(learning_rate))
    print('Epochs:        {}'.format(epochs))
    print('GPU mode:        {}'.format(gpu_mode))
    
    # Defining transforms for the training, validation, and testing sets
    # copying code over from part 1 of the lab
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # data_transforms = {'train': train_transforms, 'valid': valid_transforms, 'test': test_transforms}
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Loading data sets from the image folders
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    image_datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    dataloaders = {'train': trainloader, 'valid': validloader, 'test': testloader}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}
    
    train_size = len(train_data)
    valid_size = len(valid_data)
    test_size = len(valid_data)
    dataset_sizes = {'train': train_size, 'valid': valid_size, 'test': test_size}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_flowers = json.load(f)
    # a dictionary mapping the integer encoded categories to the actual names of the flowers
    
    # Set the GPU on if the user requested it and the machine has it
    if gpu_mode and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Current device: {}'.format(device))
    # Choose a pretrained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        print(model.classifier)
        num_in_features = model.classifier[0].in_features
        print(num_in_features)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        print(model.classifier)
        # setting input layer size to be the input feature size of the pretrained model
        # we will need this information to create our own classifier
        num_in_features = model.fc.in_features
    else:
        print("Unknown model, please choose 'vgg16' or 'resnet18'")
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Redesign the classifier using custom in_featuers and hidden_layers
    # 102 is carried over from the 102 flower classification problem
    # 102 could also be infered from cat_to_flowers dictionary size
    num_out_features = 102
    classifier = build_classifier(num_in_features, hidden_layers, num_out_features)
    criterion = nn.NLLLoss()
    if arch == 'resnet18':
        model.fc = classifier
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), learning_rate)
    elif arch == 'vgg16':
        model.classifier = classifier
        print(learning_rate)
        print(isinstance(learning_rate,float))
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    else:
        print("Error in assigning custom classifier to the pretrained model")
        
    print('='*5 + ' Architecture ' + '='*5)
    print('The classifier architecture:')
    print(classifier)
    
    model.to(device)
    
    # now finally training the whole thing
    # I wasn't sure if to input train data only or full data, i guess traindata here 
    # would be more approriate. All arguments are declared in the main function
    print('='*5 + ' Train ' + '='*5)
    model = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, device, epochs)
    
    ## now testing it, putting the model to eval() mode, because training is complete
    model.eval()
    accuracy = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloaders['test']):
            # images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # exponentiationg because output is log of probabilities, not necessary
            # probs = torch.exp(outputs)
            # Class with the highest probability is our predicted class
            # outputs.max takes the max value over the dimension 1 and returns 2 values(tensors) 
            # code is taken from pytorch tutorial site
            _, predicted = torch.max(outputs.data, 1)
            # check the first batch of images
            # if idx == 0:
            #    print(predicted) #the predicted class
            #    print(torch.exp(_)) # the predicted probability
            equals = predicted == labels.data
            #if idx == 0:
            #    print(equals)
            print(equals.float().mean())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            
            # an alternative that does not seem to work
            # If the model guesses correctly, then we have an equality
            # equality = (labels.data == probs.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions
            # not sure if it works
            # accuracy += equality.type_as(torch.FloatTensor()).mean()
            # print("Test accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))
    
    ## Save the checkpoint
    print('='*5 + ' Save ' + '='*5)
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cat_to_flowers = cat_to_flowers
    checkpoint = {
        'arch': arch,
        'input_layer': num_in_features,
        'output_layer': num_out_features,
        'checkpoint_hidden_layers': hidden_layers,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'gpu_mode': gpu_mode,
        'class_to_idx': model.class_to_idx,
        'cat_to_flowers': model.cat_to_flowers,
        'state_dict' : model.state_dict(),
        'opt_dict' : optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_name)
    print('Save the trained model to {}'.format(checkpoint_name))    

def build_classifier(num_in_features, hidden_layers, num_out_features):
    """Build a classifer with input, hidden, and output layer nodes
    hidden_layers: None or an integer array, e.g. [512, 256, 128]
    modified self-code on lab1 and from Wenjin Tao's example on github
    added output with LogSoftmax and dropout_const
    """
    
    classifier = nn.Sequential()
    if hidden_layers == None:
        classifier.add_module('fc0', nn.Linear(num_in_features, num_out_features))
        classifier.add_module('output', nn.LogSoftmax(dim=1))
        print(classifier)
    else:
        # prepare the mapping of layer sizes with zipping reversed and straight counts
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        # print(hidden_layers[:-1])
        # print(hidden_layers[1:])
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(0.1))
        for i, (h1, h2) in enumerate(layer_sizes):
            # print(i)
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU())
            classifier.add_module('drop'+str(i+1), nn.Dropout(0.1))
        hll = len(hidden_layers)
        classifier.add_module('fc'+str(hll), nn.Linear(hidden_layers[-1], num_out_features))
        classifier.add_module('output', nn.LogSoftmax(dim=1))    
    return classifier

def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, device, num_epochs):
    # modified based on transfer learning tutorial on Pytorch site
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1 , num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # probs = torch.exp(outputs)
                    # print("Exponentiated Log Probabilities")
                    # print(probs)
                    _, preds = torch.max(outputs, 1)
                    # print("Predictions")
                    # print(preds)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model   

if __name__ == '__main__':
    main()
    