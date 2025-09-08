# Classificador de Imagens: Cães e Gatos

## Descrição do Projeto

Este projeto é uma aplicação web simples que utiliza técnicas de **Deep Learning** para classificar imagens de cães e gatos. O objetivo principal é demonstrar a aplicação de **Transfer Learning** para a construção de um modelo de classificação de imagens eficiente, utilizando uma arquitetura de rede neural pré-treinada (VGG16) e uma interface web para interação.

## Tecnologias Utilizadas

  * **Python:** Linguagem de programação principal.
  * **Django:** Framework web para o desenvolvimento da aplicação.
  * **PyTorch:** Biblioteca de Machine Learning utilizada para carregar e inferir com o modelo.
  * **torchvision:** Pacote para visão computacional, usado para carregar o modelo VGG16.
  * **Pillow:** Biblioteca de processamento de imagens.
  * **HTML/CSS:** Para a interface de usuário.

## Funcionalidades

  * **Upload de Imagem:** O usuário pode fazer o upload de uma imagem.
  * **Classificação em Tempo Real:** A imagem é processada e o resultado da classificação (Cão ou Gato) é exibido na tela.
  * **Transfer Learning:** Utiliza um modelo VGG16 pré-treinado em um grande dataset, adaptando-o para a tarefa de classificação específica, o que permite alcançar alta precisão com menos dados e poder computacional.

## Estrutura do Projeto

'''
.
├── classificador_cachorro_gato/
│   ├── classificador/
│   │   ├── pycache/
│   │   ├── migrations/
│   │   ├── static/
│   │   ├── templates/
│   │   │   └── classificador/ 
│   │   ├── init.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── tests.py
│   │   ├── urls.py
│   │   └── views.py
│   ├── classificador_cachorro_gato/
│   │   ├── pycache/
│   │   ├── init.py
│   │   ├── asgi.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── media/   
│   ├── static/classificador/ 
│   │   ├── bixos.jpg
│   │   └── style.css
│   ├── .gitattributes
│   ├── .gitignore
│   ├── db.sqlite3
│   ├── manage.py
│   ├── modelo_vgg16.pth
│   ├── requirements.txt
│   └── treinar_modelo_vgg16.py
'''

## Como Executar o Projeto

Siga as instruções abaixo para configurar e rodar a aplicação em sua máquina local.

### Pré-requisitos

Certifique-se de que você tem o Python instalado em seu sistema.

### 1\. Clonar o Repositório

```bash
git clone https://github.com/otaviozdc/classificador_cachorro_gato.git
cd classificador_cachorro_gato
```

### 2\. Instalar as Dependências

Utilize o arquivo `requirements.txt` para instalar todas as bibliotecas necessárias.

```bash
pip install -r requirements.txt
```

### 3\. Executar o Servidor

Com as dependências instaladas, você pode rodar a aplicação web com o seguinte comando:

```bash
python manage.py runserver
```

### 4\. Acessar a Aplicação

Abra seu navegador e acesse a URL:

```
http://127.0.0.1:8000/
```

Você verá a tela de upload de imagens, onde poderá testar o classificador.

## Contato

Se você tiver alguma dúvida ou sugestão, sinta-se à vontade para entrar em contato.
