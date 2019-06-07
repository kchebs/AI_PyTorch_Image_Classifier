import argparse
import torch
from torchvision import transforms, datasets
from util import classifier_builder, model_generator

def load_transform(train_dir, valid_dir):
    # Defining transforms for the training, validation, and testing sets
    data_transforms = {}
    image_datasets = {}
    data_transforms['train'] = transforms.Compose([transforms.RandomRotation(30),
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])

    data_transforms['valid_test'] = transforms.Compose([transforms.Resize(256),
                   transforms.CenterCrop(224),
                   transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])])
# Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir,  transform=data_transforms['train'])
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms['valid_test'])

# Using the image datasets and the transforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    return train_dataloader, valid_dataloader, train_dataset.class_to_idx

def validate(model, valid_dataloader, criterion, device):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for i, (inputs, labels) in enumerate(valid_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def training(epochs, print_every, model, train_dataloader, valid_dataloader, optimizer, criterion, device):
        model.to(device)
        steps = 0
        
        for e in range(epochs):
            loss_so_far = 0 # the variable will be the loss for each batch
            model.train()
            for i, (inputs, labels) in enumerate(train_dataloader):
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zeros the gradients so that next iteration
                # has zero gradients/"starts fresh"
                optimizer.zero_grad()
                
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Calculate the total loss for 1 epoch of training
                loss_so_far += loss.item()
                
                if steps % print_every == 0:
                    model.eval()
                    
                    with torch.no_grad():
                        valid_loss, accuracy = validate(model, valid_dataloader, criterion, device)
                        
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                         "Training Loss: {:.3f}.. ".format(loss_so_far/print_every),
                         "Validation Loss: {:.3f}..".format(valid_loss/len(valid_dataloader)),
                        "Validation Accuracy: {:.3f}..".format(accuracy/len(valid_dataloader)))
                    
                    loss_so_far = 0
                    
                    model.train()
                    
def save_checkpoint(model, arch, save_dir, input_size, output_size, hidden_sizes, dropout):
    checkpoint = {'input_size': input_size,
                  'hidden_sizes': hidden_sizes,
                  'output_size': output_size,
                  'dropout': dropout,
                  'state_dict': model.classifier.state_dict(),
                  'arch': arch
                  }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    
def main(data_dir, save_dir, arch, learning_rate, hidden_sizes, epochs, gpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Defaults
    input_size = 25088
    output_size = 102
    dropout = 0
    print_every = 40
    
    train_dataloader, valid_dataloader, class_to_idx = load_transform(train_dir, valid_dir)
    
    model = model_generator(arch)
    if not model:
        print("Architecture not supported.")
        return
    
    model.class_to_idx = class_to_idx
    
    model.classifier = classifier_builder(input_size, hidden_sizes, output_size, dropout)
    
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    training(epochs, print_every, model, train_dataloader, valid_dataloader, optimizer, criterion, device)
    save_checkpoint(model, arch, save_dir, input_size, output_size, hidden_sizes, dropout)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Train an image classifier using pretrained architectures!',)
    parser.add_argument('data_dir', default='flowers')
    parser.add_argument('--save_dir', default='.')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', default='0.01', type=float)
    parser.add_argument('--hidden_sizes', default=0,type=int)
    parser.add_argument('--epochs', default='8',type=int)
    parser.add_argument('--gpu', default=True, action='store_true')
    input_args = parser.parse_args()
    
    main(input_args.data_dir, input_args.save_dir, input_args.arch,
         input_args.learning_rate, input_args.hidden_sizes, input_args.epochs, input_args.gpu)