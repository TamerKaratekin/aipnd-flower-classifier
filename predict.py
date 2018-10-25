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
from train import build_classifier
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import seaborn as sns

# testing github push 
# command line usage: python predict.py input_image checkpoint.pth --topk=4 --gpu_mode=True
# python predict.py flowers/test/17/image_03911.jpg checkpoint.pth --gpu_mode=True --top_k=5 --cat_names cat_to_name.json

def parse_args():
    parser = argparse.ArgumentParser(description="Reads an image and a checkpoints then prints the most likely image class and it's associated probability")
    parser.add_argument('input_image', action='store', help='image to be read and class guessed')
    parser.add_argument('checkpoint_path', action='store', default='checkpoint.pth', help='Reads the checkpoint and associated model')
    parser.add_argument("--top_k", action="store", dest="top_k", default=5, type=int, help = "Set to output top K guesses of the model")
    parser.add_argument('--cat_names', dest="category_names", default="cat_to_name.json", help ="File to extract category to name mappings")
    parser.add_argument('--gpu_mode', default=False, type=bool, dest="gpu_mode", help='Set the gpu mode')
    return parser.parse_args()

def process_image(imagepath):
    # tries to open the image and returns a tranformed nparray containing the image
    im = Image.open(imagepath)
    im.thumbnail([256, 256])
    width, height = im.size   # Get dimensions
    # print(im.size)
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im = im.crop((left, top, right, bottom))
    im.load()
    im.show()
    # print(im.size)
    np_image = np.array(im)
    # print(np_image.min())
    # print(np_image.max())
    # print(np_image.shape)
    np_image = np_image / 255
    # print(np_image.min())
    # print(np_image.max())
    #print(np_image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.divide(np.subtract(np_image, mean), std)
    # print(np_image)
    # print(np_image.min())
    # print(np_image.max())

    np_image = np_image.transpose(2,0,1)
    # print(np_image.shape)
    return np_image

def load_checkpoint(filepath, gpu_mode):
    if gpu_mode and torch.cuda.is_available():
        device = torch.device('cuda:0')
        checkpoint = torch.load(filepath)
        print("checkpoint loaded with gpu mode on")
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(filepath, device)
        print("checkpoint loaded with cpu mode on")

    # checkpoint = {
    #    'arch': arch
    #    'input_layer': input_size
    #    'output_layer': output_size
    #    'hidden_layers': checkpoint_hidden_layers
    #    'learning_rate': learning_rate
    #    'epochs': epochs
    #    'gpu_mode': gpu_mode
    #    'class_to_idx': model.class_to_idx,
    #    'cat_to_flowers': model.cat_to_flowers,
    #    'state_dict' : model.state_dict(),
    #    'opt_dict' : optimizer.state_dict()
    #}
    # need to reconstruct the model based on architecture type
    arch = checkpoint['arch']
    # not getting input_layer from dictionary of checkpoint, because it will be lookedup from arch model type after model creation
    output_layer = checkpoint['output_layer']
    checkpoint_hidden_layers = checkpoint['checkpoint_hidden_layers']
    # epochs won't be used for now
    # epoch =  checkpoint['epochs']

    # choose device mode
    if gpu_mode and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Choose a pretrained model and copy its number of input features
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_in_features = model.classifier[0].in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        # setting input layer size to be the input feature size of the pretrained model
        # we will need this information to create our own classifier
        num_in_features = model.fc.in_features
    else:
        print("Unknown model, please choose 'vgg16' or 'resnet18'")
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # num_out_features = 102
    num_out_features = output_layer
    hidden_layers = checkpoint_hidden_layers
    classifier = build_classifier(num_in_features, hidden_layers, num_out_features)
    criterion = nn.NLLLoss()
    if arch == 'resnet18':
        model.fc = classifier
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    elif arch == 'vgg16':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    else:
        print("Error in assigning custom classifier to the pretrained model")
    # after model creation, set it to the correct device type: gpu or cpu
    model.to(device)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.cat_to_flowers = checkpoint['cat_to_flowers']
    print("loaded layers, model_dict, opt_dict, and attributes successfully")

    return model

def predict(image_path, model, device, topk, cat_to_name):
    # not training the model so setting the mode to evel()
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    np_image = process_image(image_path)
    pth_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    # increasing image dimensions from 3 to 4 as model accepts batches of images, batch size = 1 in this case
    model_input = pth_image.unsqueeze_(0)
    output = model.forward(model_input)
    probs = torch.exp(output)
    top_probs, top_idx = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_idx = top_idx.detach().numpy().tolist()[0]
    class_to_idx = model.class_to_idx
    # use cat_to_name mapping entered by user as predict function input
    cat_to_flowers = cat_to_name
    # alternatively, one can use cat_to_flowers from the model category mappings
    # cat_to_flowers = model.cat_to_flowers


    idx_to_class = { v : k for k,v in class_to_idx.items()}
    top_labels = [idx_to_class[idx] for idx in top_idx]
    top_flowers = [cat_to_flowers[idx_to_class[idx]] for idx in top_idx]
    # print(top_labels)
    # print(top_probs)
    # print(top_idx)
    # print(top_labels)
    # print(top_flowers)
    return top_probs, top_labels, top_flowers

def main():
    args = parse_args()
    gpu_mode = args.gpu_mode
    if gpu_mode and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    checkpoint_path = args.checkpoint_path
    model = load_checkpoint(checkpoint_path, gpu_mode)
    model = model.to(device)
    image = args.input_image
    top_k = args.top_k
    custom_category_names_path = args.category_names
    # print(custom_category_names_path)
    with open(custom_category_names_path, 'r') as f:
        cat_to_flowers = json.load(f)
    # a dictionary mapping the integer encoded categories to the actual names of the flowers
    # with open('cat_to_name.json', 'r') as f:
    # a dictionary mapping the integer encoded categories to the actual names of the flowers
    # cat_to_flowers = json.load(f)

    top_probs, top_labels, top_flowers = predict(image, model, device, top_k, cat_to_flowers)
    print(top_probs)
    print(top_flowers)

if __name__ == '__main__':
    main()
