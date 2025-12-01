# Usage

## Setup
```bash
conda env create -f environment.yml
conda activate federated_learning
```

## Training
### 1. Vision Datasets (MNIST, CIFAR-10, SVHN)
Run the training script with the corresponding configuration file.
Replace {scenario} and {dataset_name}.
```python train_{scenario}.py --config ./conf/{scenario}/{dataset_name}.yaml```

### 2. NLP Dataset (IMDB)
```python imdb_{scenario}.py```

### 3. DarkNet
```python darknet_{scenario}.py```
