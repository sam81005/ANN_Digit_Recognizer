import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.datasets import fetch_openml
from PIL import Image, ImageDraw, ImageOps
import os
import time
# Import SciPy for advanced image processing
from scipy.ndimage import center_of_mass

# =============================================================================
# PART 1: The Improved Neural Network (No changes here)
# =============================================================================

class NeuralNetworkImproved:
    """
    An improved feedforward neural network with:
    - Deeper architecture (2 hidden layers)
    - Adam optimizer
    - Dropout regularization
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.W1 = np.random.randn(hidden_size1, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size1, 1))
        self.W2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((hidden_size2, 1))
        self.W3 = np.random.randn(output_size, hidden_size2) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((output_size, 1))
        self.m, self.v = {}, {}
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        for i in range(1, 4):
            self.m[f'dW{i}'], self.m[f'db{i}'] = np.zeros_like(getattr(self, f'W{i}')), np.zeros_like(getattr(self, f'b{i}'))
            self.v[f'dW{i}'], self.v[f'db{i}'] = np.zeros_like(getattr(self, f'W{i}')), np.zeros_like(getattr(self, f'b{i}'))

    def _relu(self, Z): return np.maximum(0, Z)
    def _relu_derivative(self, Z): return Z > 0
    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward_propagation(self, X, dropout_rate=0.5, training=True):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self._relu(Z1)
        D1 = 1
        if training:
            D1 = np.random.rand(A1.shape[0], A1.shape[1]) > dropout_rate
            A1 = A1 * D1 / (1.0 - dropout_rate)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self._relu(Z2)
        D2 = 1
        if training:
            D2 = np.random.rand(A2.shape[0], A2.shape[1]) > dropout_rate
            A2 = A2 * D2 / (1.0 - dropout_rate)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = self._softmax(Z3)
        self.cache = {'Z1': Z1, 'A1': A1, 'D1': D1, 'Z2': Z2, 'A2': A2, 'D2': D2, 'Z3': Z3, 'A3': A3}
        return A3

    def _one_hot(self, Y, num_classes):
        one_hot_Y = np.zeros((Y.size, num_classes))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def backward_propagation(self, X, Y, dropout_rate=0.5):
        m = X.shape[1]
        Y_one_hot = self._one_hot(Y, self.W3.shape[0])
        dZ3 = self.cache['A3'] - Y_one_hot
        dW3 = (1 / m) * dZ3.dot(self.cache['A2'].T)
        db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
        dA2 = self.W3.T.dot(dZ3) * self.cache['D2'] / (1.0 - dropout_rate)
        dZ2 = dA2 * self._relu_derivative(self.cache['Z2'])
        dW2 = (1 / m) * dZ2.dot(self.cache['A1'].T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = self.W2.T.dot(dZ2) * self.cache['D1'] / (1.0 - dropout_rate)
        dZ1 = dA1 * self._relu_derivative(self.cache['Z1'])
        dW1 = (1 / m) * dZ1.dot(X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        self.grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

    def update_parameters_adam(self, t, learning_rate):
        for i in range(1, 4):
            self.m[f'dW{i}'] = self.beta1 * self.m[f'dW{i}'] + (1 - self.beta1) * self.grads[f'dW{i}']
            self.m[f'db{i}'] = self.beta1 * self.m[f'db{i}'] + (1 - self.beta1) * self.grads[f'db{i}']
            self.v[f'dW{i}'] = self.beta2 * self.v[f'dW{i}'] + (1 - self.beta2) * np.square(self.grads[f'dW{i}'])
            self.v[f'db{i}'] = self.beta2 * self.v[f'db{i}'] + (1 - self.beta2) * np.square(self.grads[f'db{i}'])
            m_corr_dw, m_corr_db = self.m[f'dW{i}']/(1-self.beta1**t), self.m[f'db{i}']/(1-self.beta1**t)
            v_corr_dw, v_corr_db = self.v[f'dW{i}']/(1-self.beta2**t), self.v[f'db{i}']/(1-self.beta2**t)
            getattr(self, f'W{i}')[:] -= learning_rate * m_corr_dw / (np.sqrt(v_corr_dw) + self.epsilon)
            getattr(self, f'b{i}')[:] -= learning_rate * m_corr_db / (np.sqrt(v_corr_db) + self.epsilon)

    def get_predictions(self, A3): return np.argmax(A3, axis=0)
    def get_accuracy(self, predictions, Y): return np.sum(predictions == Y) / Y.size

    def train(self, X_train, Y_train, epochs, learning_rate, batch_size):
        t = 0
        for i in range(epochs):
            permutation = np.random.permutation(X_train.shape[1])
            X_shuffled, Y_shuffled = X_train[:, permutation], Y_train[permutation]
            for j in range(0, X_train.shape[1], batch_size):
                t += 1
                X_batch, Y_batch = X_shuffled[:, j:j+batch_size], Y_shuffled[j:j+batch_size]
                self.forward_propagation(X_batch, training=True)
                self.backward_propagation(X_batch, Y_batch)
                self.update_parameters_adam(t, learning_rate)
            if i % 10 == 0 or i == epochs - 1:
                A3_full = self.forward_propagation(X_train, training=False)
                print(f"Epoch {i}: Training Accuracy = {self.get_accuracy(self.get_predictions(A3_full), Y_train) * 100:.2f}%")

    def save_parameters(self, file_path):
        np.savez(file_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)
        print(f"Model parameters saved to {file_path}")

    def load_parameters(self, file_path):
        data = np.load(file_path)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = data['W1'], data['b1'], data['W2'], data['b2'], data['W3'], data['b3']
        print(f"Model parameters loaded from {file_path}")

# =============================================================================
# PART 2: The GUI Application (With FINAL prediction logic)
# =============================================================================

class DigitRecognizerApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.title("Final High-Accuracy Recognizer")
        self.geometry("600x400")
        self.canvas = tk.Canvas(self, width=280, height=280, bg="black", cursor="crosshair")
        self.canvas.grid(row=0, column=0, pady=10, padx=10, rowspan=3)
        self.prediction_label = tk.Label(self, text="Draw a digit...", font=("Helvetica", 24, "bold"))
        self.prediction_label.grid(row=0, column=1, padx=20, pady=10)
        self.chart_canvas = tk.Canvas(self, width=250, height=280, bg="white")
        self.chart_canvas.grid(row=1, column=1, padx=20)
        button_frame = tk.Frame(self)
        button_frame.grid(row=2, column=1, pady=10)
        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict_digit, font=("Helvetica", 14))
        self.predict_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas, font=("Helvetica", 14))
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        
    def paint(self, event):
        x, y = int(event.x), int(event.y)
        # --- IMPROVEMENT: Use a thicker brush ---
        brush_size = 15
        x1, y1 = (x - brush_size), (y - brush_size)
        x2, y2 = (x + brush_size), (y + brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="black")
        self.prediction_label.config(text="Draw a digit...")
        self.chart_canvas.delete("all")

    def predict_digit(self):
        """
        Processes the hand-drawn image using a more robust pipeline to match MNIST format.
        THIS IS THE FINAL, MOST ACCURATE VERSION.
        """
        try:
            # Convert to NumPy array for processing
            img_array = np.array(self.image)
            
            # Find bounding box
            rows = np.any(img_array, axis=1)
            cols = np.any(img_array, axis=0)
            if not np.any(rows) or not np.any(cols):
                self.prediction_label.config(text="Draw something!")
                return
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop to bounding box
            cropped = img_array[rmin:rmax+1, cmin:cmax+1]
            
            # --- IMPROVEMENT: Center using Center of Mass ---
            # Resize with padding to maintain aspect ratio and prevent distortion
            h, w = cropped.shape
            target_size = 20 # The digit should be roughly 20x20 pixels
            
            if w > h:
                new_w, new_h = target_size, int(target_size * h / w)
            else:
                new_w, new_h = int(target_size * w / h), target_size
            
            resized = np.array(Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS))
            
            # Create a 28x28 box and place the resized digit based on its center of mass
            box = np.zeros((28, 28))
            com = center_of_mass(resized)
            
            # Calculate paste coordinates
            dx = int(14 - com[1])
            dy = int(14 - com[0])
            
            # Get the slice coordinates, ensuring they are within bounds
            r_start, c_start = max(0, dy), max(0, dx)
            r_end, c_end = min(28, dy + new_h), min(28, dx + new_w)
            
            rr_start, cc_start = max(0, -dy), max(0, -dx)
            rr_end, cc_end = rr_start + (r_end - r_start), cc_start + (c_end - c_start)

            box[r_start:r_end, c_start:c_end] = resized[rr_start:rr_end, cc_start:cc_end]

            # Normalize and reshape for the model
            img_normalized = box / 255.0
            img_final = img_normalized.reshape(784, 1)
            
            # Get prediction from the model (in prediction mode)
            probabilities = self.model.forward_propagation(img_final, training=False)
            prediction = self.model.get_predictions(probabilities)
            
            self.prediction_label.config(text=f"Prediction: {prediction[0]}")
            self.update_chart(probabilities)

        except Exception as e:
            messagebox.showerror("Error", f"Could not predict: {e}")

    def update_chart(self, probabilities):
        self.chart_canvas.delete("all")
        bar_width, max_height = 25, 270
        for i, prob_scalar in enumerate(probabilities.flatten()):
            x0, y0_float = i * bar_width, max_height - (prob_scalar * max_height)
            x1, y1 = (i + 1) * bar_width, max_height
            x0_int, y0_int, x1_int, y1_int = int(x0), int(y0_float), int(x1), int(y1)
            color = "green" if i == np.argmax(probabilities) else "blue"
            self.chart_canvas.create_rectangle(x0_int, y0_int, x1_int, y1_int, fill=color)
            self.chart_canvas.create_text(x0_int + bar_width / 2, max_height + 10, text=str(i))

# =============================================================================
# PART 3: Main Execution Logic (No changes here)
# =============================================================================

def train_model_if_needed(model, file_path):
    if os.path.exists(file_path):
        print("Found existing improved model parameters. Loading...")
        model.load_parameters(file_path)
    else:
        print("No pre-trained improved model found. Starting training...")
        mnist = fetch_openml('mnist_74', version=1, as_frame=False, parser='auto')
        X, y = mnist['data'], mnist['target']
        y = y.astype(np.uint8)
        X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
        X_train, X_test = X_train.T / 255.0, X_test.T / 255.0
        start_time = time.time()
        model.train(X_train, y_train, epochs=50, learning_rate=0.002, batch_size=128)
        print(f"Training finished in {time.time() - start_time:.2f} seconds.")
        test_probs = model.forward_propagation(X_test, training=False)
        test_accuracy = model.get_accuracy(model.get_predictions(test_probs), y_test)
        print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
        model.save_parameters(file_path)

if __name__ == "__main__":
    INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE = 784, 256, 128, 10
    PARAMS_FILE = "model_parameters_improved.npz"
    nn_model = NeuralNetworkImproved(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
    train_model_if_needed(nn_model, PARAMS_FILE)
    app = DigitRecognizerApp(nn_model)
    app.mainloop()
