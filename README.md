# 🌺 Iris Flower Classification using Machine Learning

This project classifies **Iris flowers** into three species using a deep learning model built with Keras.

## 📊 Dataset Overview

The [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) is a classic dataset in pattern recognition. It contains 150 records with 4 features per sample:

- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

### 🌼 Classes

The model predicts one of the following Iris flower species:

- Iris-setosa  
- Iris-versicolor  
- Iris-virginica

## 🏗️ Model Architecture

Built using TensorFlow and Keras:

```python
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
