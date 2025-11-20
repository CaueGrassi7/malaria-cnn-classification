# Guia de Uso - Malaria Detection CNN

## 🚀 Como Executar o Projeto

### 1. Criar e Ativar Ambiente Virtual

**⚠️ IMPORTANTE**: Python 3.13+ requer uso de ambiente virtual!

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
# No macOS/Linux:
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Kaggle API (Obrigatório)

Para baixar o dataset automaticamente, configure suas credenciais do Kaggle:

1. Crie uma conta em [Kaggle](https://www.kaggle.com/)
2. Vá em **Account** → **API** → **Create New API Token**
3. Isso baixará um arquivo `kaggle.json`
4. Coloque o arquivo no diretório correto:

**Linux/Mac:**

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**

```bash
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\
```

### 3. Executar o Notebook

```bash
jupyter notebook malaria_detection.ipynb
```

Ou use o Google Colab:

1. Faça upload do notebook para o Google Drive
2. Abra com Google Colab
3. Faça upload do seu `kaggle.json` quando solicitado

### 4. Executar Todas as Células

O notebook está organizado em 16 seções:

1. **Importação de bibliotecas** - Verifica versões do TensorFlow
2. **Configuração** - Define hiperparâmetros
3. **Download do dataset** - Baixa automaticamente via Kaggle API
4. **Análise exploratória** - Visualiza amostras do dataset
5. **Pré-processamento** - Configura generators com data augmentation
6. **Arquitetura CNN** - Constrói o modelo conforme o paper
7. **Compilação** - Configura otimizador e métricas
8. **Callbacks** - EarlyStopping, ModelCheckpoint, etc.
9. **Treinamento** - Treina por até 15 épocas
10. **Visualização do histórico** - Plots de loss e accuracy
11. **Avaliação** - Calcula métricas no conjunto de teste
12. **Matriz de confusão** - Visualização detalhada
13. **Relatório de classificação** - Métricas por classe
14. **Visualização de predições** - Exemplos de predições
15. **Salvar modelo** - Exporta modelos e resultados
16. **Resumo final** - Conclusões e comparação com o paper

## 📊 Resultados Esperados

- **Acurácia**: ~97% (conforme reportado no paper)
- **Training time**: ~10-30 minutos (dependendo do hardware)
- **Dataset size**: 27.558 imagens (1.3 GB)

## 🎯 Arquivos Gerados

Após a execução completa, os seguintes arquivos serão criados em `models/`:

- `best_model.h5` - Melhor modelo durante o treinamento
- `malaria_cnn_final.h5` - Modelo final completo
- `malaria_cnn_weights.h5` - Apenas os pesos
- `training_history.png` - Gráficos de treinamento
- `confusion_matrix.png` - Matriz de confusão
- `sample_predictions.png` - Exemplos de predições
- `model_architecture.png` - Visualização da arquitetura
- `classification_report.txt` - Relatório detalhado
- `training_history.json` - Histórico em JSON
- `final_metrics.json` - Métricas finais em JSON

## 💡 Dicas

### Treinar Mais Rápido

- Use GPU: O TensorFlow detectará automaticamente
- Reduza o batch size se ficar sem memória
- Use Google Colab com GPU gratuita

### Melhorar Resultados

- Aumente o número de épocas (cuidado com overfitting)
- Experimente diferentes data augmentations
- Ajuste o learning rate
- Teste diferentes arquiteturas

### Troubleshooting

**Erro ao baixar dataset:**

- Verifique se o `kaggle.json` está configurado corretamente
- Baixe manualmente de: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

**Out of Memory:**

- Reduza o BATCH_SIZE (linha de configuração)
- Use imagens menores (já estamos usando 50x50)
- Feche outros programas

**Acurácia baixa:**

- Verifique se o dataset foi carregado corretamente
- Certifique-se de que todas as 27.558 imagens estão presentes
- Execute todas as células em ordem

## 📚 Referências

- **Paper**: "Efficient deep learning-based approach for malaria detection using red blood cell smears" - Scientific Reports, 2024
- **Dataset**: [Kaggle - Malaria Cell Images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **Framework**: TensorFlow/Keras

## 🤝 Contribuindo

Este projeto é educacional. Sinta-se livre para:

- Experimentar diferentes arquiteturas
- Adicionar novos augmentations
- Implementar ensemble de modelos
- Testar transfer learning

---

**Implementado conforme especificações do paper científico mencionado.**
