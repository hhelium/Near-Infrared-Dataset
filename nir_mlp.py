import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import to_categorical

# Configuration
batch_size = 64
num_classes = 6
epochs = 50

# Load data
X = np.loadtxt("data.csv", delimiter=",")
Y = np.loadtxt("label.csv", delimiter=",")

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

# Standardize features
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Keep a copy of y_test for manual accuracy check later
y_test1 = copy.deepcopy(y_test)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define model
def mlp():
    model = Sequential()
    model.add(Dense(331, activation='relu', input_shape=(331,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Create and compile model
model = mlp()
model.summary()
adam = Adam(learning_rate=0.0001)  # Use `learning_rate` instead of `lr` in TF2
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Predict on new data
X_p = np.loadtxt("test_data.csv", delimiter=",")
Y_p = np.loadtxt("test_label.csv", delimiter=",")
X_p = scaler.transform(X_p)

# Use argmax on predictions since `predict_classes` is deprecated
predictions = np.argmax(model.predict(X_p), axis=1)

# Accuracy check
correct = np.sum(predictions == Y_p)
total = len(predictions)

print(f"Total predictions: {total}")
print("Predicted labels:", predictions)
print("Correct predictions:", correct)
print(f"Prediction accuracy: {correct / total:.4f}")
