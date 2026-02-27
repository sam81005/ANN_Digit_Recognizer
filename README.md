# 🧠 ANN Digit Recognizer

An Artificial Neural Network (ANN) built from scratch using NumPy to recognize handwritten digits (0–9).
This project demonstrates forward propagation, backpropagation, gradient descent optimization, and model evaluation without using high-level deep learning frameworks.

---

## 📌 Project Overview

This project implements a fully connected feedforward neural network for digit classification.
The model is trained to classify handwritten digits into 10 categories (0–9).

It includes:

* Manual implementation of forward propagation
* Backpropagation algorithm
* Cross-entropy loss
* Gradient descent optimization
* Model parameter saving/loading

---

## 🛠️ Tech Stack

* Python 3
* NumPy
* Matplotlib (for visualization, if used)

---

## 📂 Project Structure

```
ANN_Proj/
│
├── main.py                         # Main training & evaluation script
├── model_parameters.npz            # Saved trained parameters
├── model_parameters_improved.npz   # Improved trained model
├── ann_report.docx                 # Project documentation
├── report.docx                     # Detailed explanation
├── expl.pdf                        # Additional explanation
└── README.md
```

---

## 🧮 Model Architecture

Example architecture:

Input Layer → Hidden Layer(s) → Output Layer

* Input Layer: 784 neurons (for 28x28 images)
* Hidden Layer(s): Fully connected layer(s)
* Output Layer: 10 neurons (Softmax activation)

---

## ⚙️ How It Works

### 1️⃣ Forward Propagation

* Weighted sum calculation
* Activation function (ReLU / Sigmoid)
* Softmax for final output

### 2️⃣ Loss Function

* Cross-Entropy Loss

### 3️⃣ Backpropagation

* Compute gradients
* Update weights using Gradient Descent

---

## 🚀 How to Run

Clone the repository:

```bash
git clone git@github.com:sam81005/ANN_Digit_Recognizer.git
cd ANN_Proj
```

Run the model:

```bash
python main.py
```

---

## 📊 Results

* Model trained using gradient descent
* Accuracy improves with better hyperparameter tuning
* Saved trained weights for reuse

---

## 📈 Future Improvements

* Add dropout for regularization
* Add more hidden layers
* Implement mini-batch gradient descent
* Add visualization of training curves
* Deploy as a web app (Flask/Streamlit)

---

## 🎯 Learning Outcomes

Through this project, I gained hands-on experience with:

* Neural network mathematics
* Backpropagation derivation
* Weight initialization techniques
* Overfitting & optimization challenges
* Saving and loading trained models

---

## 👨‍💻 Author

**Samarth Karmakar**
B.Tech Computer Science
Machine Learning & AI Enthusiast

GitHub: [https://github.com/sam81005](https://github.com/sam81005)
