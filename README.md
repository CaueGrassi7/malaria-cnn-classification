# Malaria Detection using CNN

Implementação do pipeline de classificação de imagens para detecção de malária baseado no artigo **"Efficient deep learning-based approach for malaria detection using red blood cell smears"** (Scientific Reports, 2024).

## 📋 Descrição

Este projeto implementa uma Rede Neural Convolucional (CNN) para classificar células sanguíneas em parasitadas (malária positivo) ou não infectadas, utilizando o dataset público "Malaria Cell Images Dataset" do Kaggle.

### Resultados Esperados

- **Acurácia**: ~97% (conforme reportado no paper)
- **Dataset**: 27.558 imagens (Parasitized/Uninfected)
- **Tamanho das imagens**: 50×50×3 pixels

## 🏗️ Arquitetura do Modelo

A CNN implementada segue as especificações do paper:

- **3 blocos convolucionais**:

  - Conv2D (32, 64, 128 filtros) + ReLU
  - MaxPooling2D (2×2)
  - BatchNormalization
  - Dropout (0.25)

- **Camadas densas**:
  - Flatten
  - Dense (128 neurônios) + ReLU + Dropout (0.5)
  - Dense (1 neurônio) + Sigmoid

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone <repository-url>
cd malaria-cnn-classification
```

### 2. Crie e ative um ambiente virtual (Python 3.13+)

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
# No macOS/Linux:
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure a API do Kaggle

Para baixar o dataset automaticamente, você precisa configurar suas credenciais do Kaggle:

1. Crie uma conta no [Kaggle](https://www.kaggle.com/)
2. Vá em "Account" → "API" → "Create New API Token"
3. Isso baixará um arquivo `kaggle.json`
4. Coloque o arquivo no local apropriado:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
5. Configure as permissões (Linux/Mac):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 📊 Uso

### Executar o notebook completo

```bash
jupyter notebook malaria_detection.ipynb
```

O notebook contém todas as etapas:

1. Download e organização do dataset
2. Pré-processamento das imagens
3. Construção da arquitetura CNN
4. Treinamento do modelo
5. Avaliação e visualização dos resultados

### Estrutura do Projeto

```
malaria-cnn-classification/
├── malaria_detection.ipynb    # Notebook principal
├── requirements.txt            # Dependências Python
├── README.md                   # Documentação
├── data/                       # Dataset (criado automaticamente)
│   └── cell_images/
│       ├── Parasitized/
│       └── Uninfected/
└── models/                     # Modelos salvos (opcional)
```

## 🔬 Metodologia

### Pré-processamento

- Redimensionamento: 50×50×3
- Normalização: [0, 1]
- Split: 80% treino / 20% teste
- Data augmentation: flips horizontal e vertical

### Treinamento

- **Otimizador**: Adam (lr=0.0001)
- **Loss**: Binary Crossentropy
- **Batch size**: 64
- **Epochs**: 15
- **Callback**: Early Stopping (patience=3)

### Métricas

- Acurácia
- Precisão
- Recall
- F1-Score
- Matriz de Confusão

## 📈 Resultados

Os resultados incluem:

- Curvas de loss e acurácia (treino vs validação)
- Matriz de confusão
- Relatório de classificação completo
- Comparação com os resultados do paper (~97% accuracy)

## 🔗 Referências

- **Paper**: "Efficient deep learning-based approach for malaria detection using red blood cell smears" - Scientific Reports, 2024
- **Dataset**: [Malaria Cell Images Dataset - Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

## 📝 Licença

Este projeto é para fins educacionais e de pesquisa.

## 👥 Autor

Implementado seguindo as especificações do paper científico mencionado.
