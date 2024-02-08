import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from simpledataloader import SimpleDataLoader
from torchvision.datasets import ImageFolder
from model import VGG16
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
import torchvision.models.quantization as models
data_path = '/home/naserwin/sarisa/dogsvscats'


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    
])

transform_valid = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
])
dataset_train = ImageFolder('/home/naserwin/sarisa/dogsvscats/train', transform=transform_train)

dataset_valid = ImageFolder('/home/naserwin/sarisa/dogsvscats/valid', transform=transform_valid)

train_loader = SimpleDataLoader(data=dataset_train, labels=dataset_train.targets, batch_size=16, shuffle=True)
valid_loader = SimpleDataLoader(data=dataset_valid, labels=dataset_valid.targets, batch_size=16, shuffle=False)
vgg16 = VGG16(in_channels=3, num_classes=2)



criterion = nn.CrossEntropyLoss()
optimizer = Adam(vgg16.parameters(), lr=0.001)

vgg16.train()

batch_size = 16
num_epochs =  7
best_accuracy = 0.0  
best_model_weights = None 
import torch

save_interval = 1  

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images, labels
        
        outputs = vgg16(images)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            accuracy = total_correct / total_samples
            print('Epoch [%2d/%2d], Batch [%3d/%3d], Loss: %.4f, Accuracy: %.4f'
                  % (epoch + 1, num_epochs, batch_idx + 1, train_loader.num_batches, loss.item(), accuracy))
    
    epoch_accuracy = total_correct / total_samples
    
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_model_weights = vgg16.state_dict().copy()

    if (epoch + 1) % save_interval == 0:
        torch.save(vgg16.state_dict(), '/home/naserwin/sarisa/dogsvscats/weight/checkpoint_epoch{}.pth'.format(epoch + 1))

torch.save(best_model_weights, '/home/naserwin/sarisa/dogsvscats/weight/best_model_weights.pth')




vgg16.load_state_dict(torch.load('/home/naserwin/sarisa/dogsvscats/weight/best_model_weights.pth'))
vgg16= torch.quantization.quantize_dynamic(
    vgg16,  
    {nn.Conv2d, nn.Linear},  
    dtype=torch.qint8  
)

vgg16.eval()

criterion = nn.CrossEntropyLoss()


total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in valid_loader:
        
        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        
        
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()


accuracy = total_correct / total_samples
print(f'accuracy: {accuracy:.4f}')





