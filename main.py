import numpy as np
from torch.utils.data import DataLoader 
from torchvision import datasets 
from torchvision.transforms import ToTensor 
import matplotlib.pyplot as plt 
import torch
from torch import save, load
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
batch_size = 64
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_batches = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_batches = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

dataiter = iter(train_batches)
images, labels = next(dataiter)

labels_dict = [
  "T-shirt/Top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle Boot"]

print(f"Unique labels for FashionMNIST: {np.unique(labels, return_counts=True)}")
print(f"Number of images for training: {len(trainset)}")
print(f"Number of created batches for training: {len(train_batches)}") 
print(f"Number of images for testing: {len(testset)}")
print(f"Number of created batches for testing: {len(test_batches)}")

#UNDERSTAND DATA
for batch in train_batches:        
    print(f"Type of 'batch': {type(batch)}")         
    data, labels = batch       
    print(f"Type of 'data': {type(data)}")     
    print(f"Type of 'labels': {type(labels)}")       
    print(f"Shape of data Tensor from current batch: {data.shape}")     
    print(f"Shape of first image Tensor from current batch: {data[0].shape}")     
    print(f"Shape of labels Tensor from current batch: {labels.shape}")     
    print(f"Labels included in the 'labels' Tensor from current batch: {labels}")
    break


plt.figure(figsize=(10,10)) 
for batch in train_batches: 
    data, labels = batch 
    for i in range(25): 
        plt.subplot(5, 5, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap="gray", interpolation="none")
        plt.xlabel(f"{labels[i]} - {labels_dict[labels[i]]}", fontsize = 11) 
        plt.xticks([]) 
        plt.yticks([]) 
    plt.savefig("Sample_Images_Labels.jpg")
    break

'''
Define the Convolutional Neural Network Arquitecture
'''
#CNN Fashion MNIST Classifier
class DigitClassifier(torch.nn.Module):
    #Class Constructor
    def __init__(self):
        #Parent Class Constructor
        super().__init__()
        self.model = torch.nn.Sequential(
            #Feature Extraction Stage
            #Value of each output_channels the same as next in_channels value
            #in_channels = 1 -> Grayscale Images
            #Conv2d: in_channels, out_channels, kernel_size  
            torch.nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=(kernel_dim, kernel_dim)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=(kernel_dim, kernel_dim)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=(kernel_dim, kernel_dim)),
            torch.nn.ReLU(),
            #Classification Stage
            torch.nn.Flatten(),
            #Linear: in_features, out_features
            torch.nn.Linear(in_features=in_features, out_features=10)
        )
    def forward(self, x):
        return self.model(x)

out_1 = 32
out_2 = 64
out_3 = 128
kernel_dim = 3
in_features = 128*22*22

'''
TRAINING PROCEDURE 
'''

#Instance of the Neural Network, Loss & Optimizer 
cnn = DigitClassifier()
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3) 
loss_fn = torch.nn.CrossEntropyLoss() 

#Number of times to pass through every batch 
num_epochs = 5
#Set the CNN to training mode 
cnn.train(True)  
for epoch in range(num_epochs):      
    for batch_idx, (X, y) in enumerate(train_batches):         
        #Pass the batch of images to obtain a prediction         
        y_pred = cnn(X)         
        #Compute the loss comparing the predictions made by the CNN with the original labels         
        loss = loss_fn(input=y_pred, target=y)          
        #Perform backpropagation with the computed loss to compute the gradients         
        loss.backward()          
        #Update the weights with regard to the computed gradients to minimize the loss         
        opt.step()          
        #In each iteration we want to compute new gradients, that is why we set the gradients to 0         
        opt.zero_grad()          
        #Print to check the progress         
        if batch_idx % 50 == 0:             
            print(f"Train Epoch: {epoch} [{ batch_idx *len(data)}/{len(train_batches.dataset)} ({100.0 * batch_idx / len(train_batches):.0f}%)]\tLoss: {loss.item():.6f}"             
                )      
    print(f"Epoch: {epoch} loss is {loss.item()}") 
 
#Set the model to evaluation mode to predict data  
cnn.eval() 
with torch.no_grad():  #Avoid calculating gradients 
    #Predict the images in batches 
    for images, labels in test_batches: 
        #Predict the images with probabilities 
        test_output = cnn(images) 
        #For each image, take the highest probability 
        y_test_pred = torch.max(test_output, 1)[1].data.squeeze() 
        #Measure the accuracy 
        accuracy = (y_test_pred == labels).sum().item() / float(labels.size(0)) 
print("VALIDATION SET ACCURACY: %.2f" % accuracy) 

#Predict on various images of the dataset
img_array = [155, 258, 552, 789, 900]

for img_id in img_array:
    img_tensor, label = testset[img_id]
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        img_pred = cnn(img_tensor) 
        probabilities = torch.softmax(img_pred, dim=1).squeeze()
        pred_label = torch.argmax(probabilities).item()

        if label == pred_label:
            color_greenORred = "green" 
        else:
            color_greenORred = "red"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"Ground Truth: {labels_dict[label]} - Prediction: {labels_dict[pred_label]} ({probabilities[pred_label]*100:.1f}%)", color = color_greenORred)

        ax1.imshow(img_tensor[0, 0, :, :], cmap="gray")
        ax1.axis("off")
        ax1.set_title(f"Label: {labels_dict[label]}")

        ax2.bar(range(len(labels_dict)), probabilities.numpy(), color=color_greenORred)
        ax2.set_xticks(range(len(labels_dict)))
        ax2.set_xticklabels(labels_dict, rotation=45, ha="right")
        ax2.set_title("Probability for each class")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Probability")

        plt.tight_layout()
        plt.savefig(f"./prediction_of_test_image_{img_id}.png")