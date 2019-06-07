import torch
from torchvision import models
from collections import OrderedDict

def model_generator(arch):
    SUPPORTED_ARCHS = ['vgg13', 'vgg16', 'vgg19']
    
    if arch in SUPPORTED_ARCHS:
        exec_string = 'models.' + arch + '(pretrained=True)'
    else:
        return
    
    model = eval(exec_string)
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def classifier_builder(input_size, hidden_sizes, output_size, dropout):
    if hidden_sizes == 0:
        hidden_sizes = []
    else:
        hidden_sizes = [hidden_sizes]
    sizes = [input_size] + hidden_sizes + [output_size]
    layer_list = []
    
    for i in range(1, len(sizes)):
        layer_list.append(('fc' + str(i), torch.nn.Linear(sizes[i-1], sizes[i])))
        if i < len(sizes) - 1:
            layer_list.append(('relu' + str(i), torch.nn.ReLU()))
            layer_list.append(('drop' + str(i), torch.nn.Dropout(dropout)))
            
    layer_list.append(('output', torch.nn.LogSoftmax(dim=1)))
    classifier = torch.nn.Sequential(OrderedDict(layer_list))
    
    return classifier