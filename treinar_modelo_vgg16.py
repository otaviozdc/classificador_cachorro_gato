import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# Preparação dos dados de imagem (VGG16 - 224x224)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Caminhos para os dados
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'dados')
train_dir = os.path.join(data_dir, 'train')

# Carregar os dados de treinamento
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Carregar o modelo VGG16 pré-treinado
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# Congelar os parâmetros das camadas convolucionais
for param in model.parameters():
    param.requires_grad = False

# Substituindo a camada de saída (gato/cachorro)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)

# Definindo a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
print("Iniciando o treinamento do modelo VGG16...")
num_epochs = 5 
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Época [{epoch+1}/{num_epochs}], Perda: {loss.item():.4f}")

# Salvar o novo modelo treinado
model_path = os.path.join(script_dir, 'modelo_vgg16.pth')
torch.save(model.state_dict(), model_path)
print(f"Modelo VGG16 salvo em '{model_path}'")