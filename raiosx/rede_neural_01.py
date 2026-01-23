import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

# Redimensionamento, normalização e conversão para tensor

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
# Carregamento do dataset de imagens
dataset = datasets.ImageFolder('./data', transform=transform)

# Divisão do dataset em treino e validação
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders para treino e validação
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#verificação das imagens carregadas
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
plt.show()

# Definição da arquitetura da rede neural

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instanciação do modelo, definição da função de perda e do otimizador
def treino_modelo(modelo, train_loader, val_loader, criterio, otimizador, num_epochs=10):

    for epoch in range(num_epochs):
        modelo.train()
        inicio = time()
        perda_treino = 0.0
        for imagens, etiquetas in train_loader:
            imagens, etiquetas = imagens.to(dispositivo), etiquetas.to(dispositivo)
            otimizador.zero_grad() #zera os gradientes
            saidas = modelo(imagens)
            perda = criterio(saidas, etiquetas)
            perda.backward()
            otimizador.step()
            perda_treino += perda.item() * imagens.size(0)

        perda_treino /= len(train_loader.dataset)

        modelo.eval()
        perda_val = 0.0
        with torch.no_grad():
            for imagens, etiquetas in val_loader:
                imagens, etiquetas = imagens.to(dispositivo), etiquetas.to(dispositivo)
                saidas = modelo(imagens)
                perda = criterio(saidas, etiquetas)
                perda_val += perda.item() * imagens.size(0)

        perda_val /= len(val_loader.dataset)

        print(f'Época: {epoch+1}/{num_epochs}, \nPerda Treino: {perda_treino:.4f}, \nPerda Validação: {perda_val:.4f}')
        print(f'Tempo por época: {time() - inicio:.2f} segundos\n')


def validacao_modelo(modelo, val_loader):
    modelo.eval()
    acertos = 0
    total = 0
    with torch.no_grad():
        for imagens, etiquetas in val_loader:
            imagens, etiquetas = imagens.to(dispositivo), etiquetas.to(dispositivo)
            saidas = modelo(imagens)
            _, previsoes = torch.max(saidas.data, 1)
            total += etiquetas.size(0)
            acertos += (previsoes == etiquetas).sum().item()

    precisao = 100 * acertos / total
    print(f'Precisão na validação: {precisao:.2f}%')
    modelo.train()

modelo = Modelo()
criterio = nn.CrossEntropyLoss()
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(dispositivo)
otimizador = optim.Adam(modelo.parameters(), lr=0.0001)

treino_modelo(modelo, train_loader, val_loader, criterio, otimizador, num_epochs=10)
validacao_modelo(modelo, val_loader)

# Salvando o modelo treinado
torch.save(modelo.state_dict(), 'modelo_raio_x.pth')
print("Modelo salvo com sucesso!")