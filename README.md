## Project: DynamicNet with Orthogonalization

### Description

DynamicNet is a neural network with dynamic layers designed to improve the orthogonalization of output vectors. The architecture is based on an original method that includes an orthogonalization layer, allowing for higher approximation accuracy and stability in training deep networks. This project implements a procedure for orthogonalizing the representation of the hidden layer's output activity using mapping applied to all layers of the deep network except the last one.

### Key Components

1. **MappingLayer**: A special layer for orthogonalization, adjusting weights to achieve orthogonal directions of output activity.
2. **DynamicNet**: The main network consisting of multiple dynamic layers, including linear transformations, activation functions, batch normalization, and the orthogonalization layer.
3. **Training and Validation**: The training process includes loss function calculation, backpropagation, and early stopping.

### Features

- **MappingLayer**: Implements the orthogonalization method, computing increments to adjust connection weights to achieve orthogonal output vectors.
- **DynamicNet**: Allows configuration of the number of layers and the number of elements in each layer to achieve the desired approximation accuracy.
- **Early Stopping**: Prevents overfitting by stopping training if no improvement in validation loss is observed for a specified number of epochs.

### Input Data

The input data consists of pairs of tensors, where each tensor is a 2-dimensional vector. These pairs are used to train the network to approximate the transformation from input vectors to output vectors.

### Example Usage

#### Import Libraries and Generate Data

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def generate_examples(num_examples):
    # Example data generation function
    X = np.random.randn(num_examples, 2)
    Y = np.random.randn(num_examples, 2)
    return [(torch.tensor(X[i], dtype=torch.float32), torch.tensor(Y[i], dtype=torch.float32)) for i in range(num_examples)]
```


#### Metrics and Training Execution

After defining the model and training functions, we can proceed with generating the training, validation, and test data, training the model, and evaluating its performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### Visualizations

To better understand the training process and the effect of orthogonalization, visualizations of loss curves and weight distributions are provided. This helps in diagnosing issues and understanding the model's behavior during training.

### Getting Started

1. **Clone the Repository**: Start by cloning the repository to your local machine.
    ```bash
    git clone https://github.com/yourusername/DynamicNet.git
    cd DynamicNet
    ```
2. **Install Dependencies**: Ensure you have all necessary dependencies installed. You can use the provided `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Example**: Execute the example script to see DynamicNet in action.
    ```bash
    python example.py
    ```

### Input Data

The input data for this project consists of pairs of 2-dimensional vectors. Each pair represents a sample where the first vector is the input and the second vector is the target output. The model is trained to learn the mapping from input vectors to target vectors.  ![gif](IMG_20240628_230302_739.gif). 

### Project Structure
All of this below included in Vector_analysis.ipynb:
- **`DynamicNet.py`**: Contains the implementation of the DynamicNet model and MappingLayer.
- **`example.py`**: An example script demonstrating how to generate data, train the model, and evaluate its performance.
- **`requirements.txt`**: Lists all the dependencies required to run the project.
- **`README.md`**: Provides an overview and instructions for the project.

### Results

The project includes a GIF file that visually demonstrates the training process and the orthogonalization effect. [Link to GIF](repo/orthogonalization_example.gif).

### Conclusion

DynamicNet with orthogonalization offers a powerful solution for training deep neural networks with enhanced stability and approximation accuracy. This project includes a complete training cycle, from data generation to metric evaluation, providing a robust framework for experimenting with orthogonalization in neural networks.

### Future Work

Future improvements for this project could include:
- Extending the model to support other types of neural network layers.
- Experimenting with different orthogonalization techniques.
- Integrating more advanced optimization algorithms.
- Applying the model to real-world datasets to evaluate its performance in practical scenarios.

Feel free to explore, modify, and contribute to this project. Your feedback and contributions are highly appreciated!

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact

For any questions or suggestions, please contact the project maintainer at your-email@example.com.

---

By including detailed explanations and comprehensive instructions, this README aims to make it easier for users to understand and utilize DynamicNet with orthogonalization.

## Проект: DynamicNet с Ортогонализацией

### Описание

DynamicNet — это нейронная сеть с динамическими слоями, специально разработанная для улучшенной ортогонализации выходных векторов. В основе архитектуры лежит оригинальный метод, включающий слой ортогонализации, что позволяет достичь более высокой точности аппроксимации и стабильности при обучении глубоких сетей. Этот проект реализует процедуру ортогонализации представления выходной активности скрытых слоев с использованием картирования, применяемого ко всем слоям глубокой сети, кроме последнего.

### Основные компоненты

1. **MappingLayer**: Специальный слой для ортогонализации, который корректирует веса для достижения ортогональных направлений выходной активности.
2. **DynamicNet**: Основная сеть, состоящая из нескольких динамических слоев, включающих линейные преобразования, функции активации, батч-нормализацию и слой ортогонализации.
3. **Обучение и Валидация**: Процесс обучения, включающий в себя расчет функции потерь, обратное распространение ошибки и раннюю остановку.

### Функции

- **MappingLayer**: Реализует метод ортогонализации, который вычисляет приращения для корректировки весов связи, чтобы добиться ортогональности выходных векторов.
- **DynamicNet**: Позволяет настраивать количество слоев и количество элементов в каждом слое для достижения заданной точности аппроксимации.
- **Ранняя остановка**: Применяется, чтобы предотвратить переобучение, останавливая обучение, если улучшения в валидационной потере не наблюдаются в течение заданного количества эпох.

### Пример использования

#### Импорт библиотек и генерация данных

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def generate_examples(num_examples):
    # Пример функции генерации данных
    X = np.random.randn(num_examples, 2)
    Y = np.random.randn(num_examples, 2)
    return [(torch.tensor(X[i], dtype=torch.float32), torch.tensor(Y[i], dtype=torch.float32)) for i in range(num_examples)]
```

### Заключение

DynamicNet с ортогонализацией предлагает мощное решение для обучения глубоких нейронных сетей с улучшенной стабильностью и точностью аппроксимации. Этот проект включает полный цикл обучения, от генерации данных до оценки метрик
