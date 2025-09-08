from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import transforms, models

# Carrega o modelo VGG16 pré-treinado e substitui a camada final
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# Congela os parâmetros das camadas convolucionais
for param in model.parameters():
    param.requires_grad = False

# Substitui a camada de saída (gato/cachorro)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)

# Caminho para o novo modelo treinado
model_path = os.path.join(os.path.dirname(__file__), '..', 'modelo_vgg16.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")

# Carrega os pesos do modelo treinado
model.load_state_dict(torch.load(model_path))
model.eval() # Coloca o modelo em modo de avaliação

# Define a transformação para a imagem de entrada
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def upload_imagem(request):
    resultado = None
    imagem_url = None
    if request.method == 'POST' and 'imagem' in request.FILES:
        fs = FileSystemStorage()
        nome_arquivo = fs.save(request.FILES['imagem'].name, request.FILES['imagem'])
        imagem_url = fs.url(nome_arquivo)
        caminho_completo = os.path.join(fs.location, nome_arquivo)

        try:
            # Pré-processamento com PyTorch
            imagem_pil = Image.open(caminho_completo).convert('RGB')
            imagem_tensor = transform(imagem_pil).unsqueeze(0)
            
            # Predição com o modelo PyTorch
            with torch.no_grad():
                outputs = model(imagem_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
            if predicted.item() == 0:
                resultado = "Gato"
            else:
                resultado = "Cachorro"
            
        except Exception as e:
            resultado = f"Erro na predição: {e}"
        
        return render(request, 'classificador/index.html', {'resultado': resultado, 'imagem_url': imagem_url})

    return render(request, 'classificador/index.html', {'resultado': resultado})