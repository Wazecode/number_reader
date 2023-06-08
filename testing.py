import cv2 as cv
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

test = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

classes = ['0','1','2','3','4','5','6','7','8','9']

model.eval()



for img, ind in test:

    pred = model(img)
    predicted = classes[pred[0].argmax(0)]
    print(predicted, ind)

cv.waitKey(0)
cv.destroyAllWindows()
