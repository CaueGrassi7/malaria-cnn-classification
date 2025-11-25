# Malaria Detection using CNN - Estudo Comparativo

Implementação de um **estudo comparativo multi-experimento** para classificação de imagens de detecção de malária baseado no artigo **"Efficient deep learning-based approach for malaria detection using red blood cell smears"** (Scientific Reports, 2024).

## 📋 Descrição

Este projeto implementa e compara **3 configurações diferentes** de Redes Neurais Convolucionais (CNN) para classificar células sanguíneas em parasitadas (malária positivo) ou não infectadas, utilizando o dataset público "Malaria Cell Images Dataset" do Kaggle.

### Características do Dataset

- **Total**: 27.558 imagens (balanceado 50/50)
- **Classes**: Parasitized (13.779) e Uninfected (13.779)
- **Tamanho das imagens**: 50×50×3 pixels
- **Fonte**: [Kaggle - Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

### Resultados Obtidos

- **Baseline (Paper)**: 93.34% de acurácia
- **Alta Capacidade**: 94.32% de acurácia
- **Augmentation Agressivo**: 94.61% de acurácia (melhor resultado)
- **Acurácia reportada no paper**: 97.00%

## 🏗️ Arquitetura dos Modelos

O projeto implementa **3 experimentos diferentes** para comparação:

### Experimento 1: Baseline (Paper) 🎯

Replicação exata da configuração do artigo de referência:

- **3 blocos convolucionais**: Conv2D (32, 64, 128 filtros) + ReLU
- MaxPooling2D (2×2) + BatchNormalization + Dropout (0.25)
- **Camada densa**: 128 neurônios + ReLU + Dropout (0.5)
- **Saída**: 1 neurônio com Sigmoid
- **Total de parâmetros**: ~684K

### Experimento 2: Alta Capacidade 🚀

Rede com maior capacidade para testar se mais parâmetros melhoram o desempenho:

- **3 blocos convolucionais**: Conv2D (64, 128, 256 filtros) - **dobro da capacidade**
- MaxPooling2D (2×2) + BatchNormalization + Dropout (0.3)
- **Camada densa**: 256 neurônios + ReLU + Dropout (0.5)
- **Saída**: 1 neurônio com Sigmoid

### Experimento 3: Augmentation Agressivo + Regularização 🎲

Data augmentation intenso e maior regularização para melhorar generalização:

- **3 blocos convolucionais**: Conv2D (32, 64, 128 filtros) - igual ao baseline
- MaxPooling2D (2×2) + BatchNormalization + Dropout (0.4) - **maior regularização**
- **Camada densa**: 128 neurônios + ReLU + Dropout (0.6)
- **Saída**: 1 neurônio com Sigmoid

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone <repository-url>
cd malaria-cnn-classification
```

### 2. Crie e ative um ambiente virtual (Python 3.8+)

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

O notebook contém todas as etapas do estudo comparativo:

1. Download e organização do dataset
2. Análise exploratória dos dados
3. Configuração dos 3 experimentos
4. Construção das arquiteturas CNN
5. Treinamento dos 3 modelos
6. Avaliação e comparação dos resultados
7. Geração de gráficos e tabelas comparativas

### Estrutura do Projeto

```
malaria-cnn-classification/
├── malaria_detection.ipynb    # Notebook principal com estudo comparativo
├── requirements.txt            # Dependências Python
├── README.md                   # Documentação
├── data/                       # Dataset (criado automaticamente)
│   └── cell_images/
│       ├── Parasitized/
│       └── Uninfected/
├── models/                     # Modelos treinados e métricas
│   ├── baseline_paper_*         # Resultados do experimento 1
│   ├── exp2_high_capacity_*     # Resultados do experimento 2
│   ├── exp3_augmentation_*      # Resultados do experimento 3
│   └── comparative_results.csv  # Tabela comparativa
└── figures/                    # Gráficos e visualizações
    ├── *_training_curves.png    # Curvas de treinamento
    ├── *_confusion_matrix.png   # Matrizes de confusão
    └── *_comparison.png         # Gráficos comparativos
```

## 🔬 Metodologia

### Pré-processamento (Comum a todos os experimentos)

- **Redimensionamento**: 50×50×3 pixels
- **Normalização**: [0, 1] (rescale=1./255)
- **Split**: 80% treino (22.048 imagens) / 20% validação (5.510 imagens)
- **Data augmentation**: Varia por experimento (ver detalhes abaixo)

### Configurações de Treinamento por Experimento

#### Experimento 1: Baseline (Paper)

- **Otimizador**: Adam (lr=0.0001)
- **Loss**: Binary Crossentropy
- **Batch size**: 64
- **Epochs**: 15
- **Data augmentation**: Apenas flips horizontal e vertical
- **Dropout**: 0.25 (conv) / 0.5 (dense)

#### Experimento 2: Alta Capacidade

- **Otimizador**: Adam (lr=0.0001)
- **Loss**: Binary Crossentropy
- **Batch size**: 64
- **Epochs**: 20
- **Data augmentation**: Apenas flips horizontal e vertical
- **Dropout**: 0.3 (conv) / 0.5 (dense)

#### Experimento 3: Augmentation Agressivo

- **Otimizador**: Adam (lr=0.0005)
- **Loss**: Binary Crossentropy
- **Batch size**: 32
- **Epochs**: 20
- **Data augmentation**: Flips + rotação (15°) + zoom (0.1) + shifts (0.1)
- **Dropout**: 0.4 (conv) / 0.6 (dense)

### Callbacks (Comuns a todos)

- **Early Stopping**: Monitora `val_loss` com patience=3
- **Model Checkpoint**: Salva melhor modelo baseado em `val_accuracy`
- **ReduceLROnPlateau**: Reduz learning rate quando `val_loss` para de melhorar

### Métricas Avaliadas

- Acurácia (Accuracy)
- Precisão (Precision)
- Recall (Sensibilidade)
- F1-Score
- AUC (Area Under Curve)
- Matriz de Confusão

## 📈 Resultados

### Resultados por Experimento

| Experimento                | Acurácia   | Precision | Recall | F1-Score   |
| -------------------------- | ---------- | --------- | ------ | ---------- |
| **Baseline (Paper)**       | 93.34%     | 0.9070    | 0.9659 | 0.9355     |
| **Alta Capacidade**        | 94.32%     | 0.9348    | 0.9528 | 0.9437     |
| **Augmentation Agressivo** | **94.61%** | 0.9235    | 0.9728 | **0.9475** |

### Análise Comparativa

- **Melhor resultado**: Experimento 3 (Augmentation Agressivo) com 94.61% de acurácia
- **Comparação com paper**: Todos os experimentos ficaram abaixo da acurácia reportada (97%), mas com resultados consistentes e próximos
- **Insights**:
  - Aumentar a capacidade da rede (Exp 2) melhorou ligeiramente os resultados
  - Data augmentation agressivo + regularização (Exp 3) obteve o melhor desempenho geral

### Artefatos Gerados

O notebook gera automaticamente:

- **Modelos treinados**: `.h5` files para cada experimento
- **Histórico de treinamento**: JSON com métricas por época
- **Relatórios de classificação**: Text files com métricas detalhadas
- **Gráficos de treinamento**: Curvas de loss, accuracy, precision e recall
- **Matrizes de confusão**: Visualizações para cada experimento
- **Gráficos comparativos**: Comparação de acurácia, F1-score e métricas entre experimentos
- **Tabela comparativa**: CSV com todos os resultados

## 🔗 Referências

- **Paper**: "Efficient deep learning-based approach for malaria detection using red blood cell smears" - Scientific Reports, 2024
- **Dataset**: [Malaria Cell Images Dataset - Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

## 📝 Licença

Este projeto é para fins educacionais e de pesquisa.

## 🎯 Objetivos do Estudo

Este projeto foi desenvolvido para:

1. **Validar a implementação**: Replicar o baseline do paper para garantir correção
2. **Explorar variações**: Testar diferentes estratégias (capacidade vs augmentation)
3. **Comparar abordagens**: Identificar qual configuração funciona melhor
4. **Gerar insights**: Entender trade-offs entre complexidade e desempenho

## 📝 Notas Técnicas

- **Framework**: TensorFlow 2.20.0 / Keras 3.12.0
- **Reprodutibilidade**: Seeds fixos (42) para garantir resultados reproduzíveis
- **GPU**: Suporta GPU, mas funciona também em CPU
- **Tempo de treinamento**: ~15-20 minutos por experimento em CPU moderno

## 👥 Autor

Implementado como estudo comparativo baseado nas especificações do paper científico mencionado.
