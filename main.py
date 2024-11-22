from datetime import datetime
import numpy as np
from keras.src.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.layers import Input
import pandas as pd

# Wczytanie danych z pliku CSV
winedb_path = '/wine.data'
columns = ['Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
           'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
           'Color_intensity', 'Hue', 'OD280/OD315', 'Proline']

# Wczytanie pliku CSV z przypisaniem nazw kolumn
wine_db = pd.read_csv(winedb_path, header=None, names=columns)
print(wine_db)

# Przetasowanie danych
data = wine_db.sample(frac=1, random_state=42).reset_index(drop=True)

# Oddzielenie cech od etykiet
B1 = data.drop('Class', axis=1).values
B2 = data['Class'].values.reshape(-1, 1)

# One-hot encoding etykiet
encoder = OneHotEncoder(sparse_output=False)  # Zmiana z `sparse=False` na `sparse_output=False`
B2_encoded = encoder.fit_transform(B2)

# Podział na zbiór treningowy i testowy
B1_train, B1_test, B2_train, B2_test = train_test_split(B1, B2_encoded, test_size=0.2, random_state=42)

# Wyświetlenie kształtu danych, aby upewnić się, że wszystko działa poprawnie
print("Shape of X_train:", B1_train.shape) #odpowiedz .shape (liczba_wierszy, liczba_kolumn)
print("Shape of X_test:", B1_test.shape)
print("Shape of y_train:", B2_train.shape)
print("Shape of y_test:", B2_test.shape)

# Konfiguracja TensorBoard do callbacków zgodnie z zaleceniem formatu
log_dir_1 = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_model_1"
tensorboard_callback_1 = TensorBoard(log_dir=log_dir_1, histogram_freq=1)

log_dir_2 = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_model_2"
tensorboard_callback_2 = TensorBoard(log_dir=log_dir_2, histogram_freq=1)
# if os.path.exists("model_1.h5"):
#     model_1 = load_model("model_1.h5")
#     print("Model 1 wczytany z pliku.")
# else:
# Model 1 z użyciem warstwy Input
model_1 = Sequential(name="Model_1")  # Zmiana nazwy na Model_1 (bez spacji)
model_1.add(Input(shape=(B1_train.shape[1],), name="Input_Layer"))
model_1.add(Dense(64, activation='relu', kernel_initializer=HeNormal(), name="Hidden_Layer_1"))
model_1.add(Dense(32, activation='relu', kernel_initializer=HeNormal(), name="Hidden_Layer_2"))
model_1.add(Dense(3, activation='softmax', name="Output_Layer"))
model_1.save("model_1.h5")
# Kompilacja Modelu 1
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Wyświetlenie informacji o modelu
model_1.summary()

# Trening Modelu 1
epochs = 100
batch_size = 32
history_1 = model_1.fit(B1_train, B2_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(B1_test, B2_test),
                        callbacks=[tensorboard_callback_1],
                        verbose=1)
# if os.path.exists("model_2.h5"):
#     model_2 = load_model("model_2.h5")
#     print("Model 2 wczytany z pliku.")
# else:
# Model 2 z użyciem crossentropy i softmax
model_2 = Sequential(name="Model_2")  # Zmiana nazwy na Model_2 (bez spacji)
model_2.add(Input(shape=(B1_train.shape[1],), name="Input_Layer"))
model_2.add(Dense(128, activation='tanh', kernel_initializer=GlorotUniform(), name="Hidden_Layer_1"))
model_2.add(Dense(64, activation='tanh', kernel_initializer=GlorotUniform(), name="Hidden_Layer_2"))
model_2.add(Dense(32, activation='tanh', kernel_initializer=GlorotUniform(), name="Hidden_Layer_3"))
model_2.add(Dense(3, activation='softmax', name="Output_Layer"))
model_2.save("model_2.h5")
# Kompilacja Modelu 2
model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Wyświetlenie informacji o modelu
model_2.summary()

# Trening Modelu 2
history_2 = model_2.fit(B1_train, B2_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(B1_test, B2_test),
                        callbacks=[tensorboard_callback_2],
                        verbose=1)

# Ocena modeli
test_loss_1, test_acc_1 = model_1.evaluate(B1_test, B2_test, verbose=0)
print(f"Model 1 - Test Accuracy: {test_acc_1:.4f}")

test_loss_2, test_acc_2 = model_2.evaluate(B1_test, B2_test, verbose=0)
print(f"Model 2 - Test Accuracy: {test_acc_2:.4f}")


# Ocena modeli
test_loss_1, test_acc_1 = model_1.evaluate(B1_test, B2_test, verbose=0)
print(f"Model 1 - Test Accuracy: {test_acc_1:.4f}")

test_loss_2, test_acc_2 = model_2.evaluate(B1_test, B2_test, verbose=0)
print(f"Model 2 - Test Accuracy: {test_acc_2:.4f}")

best_model = model_1 if test_acc_1 > test_acc_2 else model_2

# Funkcja predykcyjna dla użytkownika
def predict_wine_category(features):
    prediction = best_model.predict(np.array([features]))
    predicted_class = np.argmax(prediction) + 1
    return predicted_class

user_input = np.array([13.2, 2.77, 2.51, 18.5, 98, 2.64, 2.43, 0.26, 1.63, 5.4, 0.94, 3.17, 680])
predicted_class = predict_wine_category(user_input)
print(f"Predicted wine category: {predicted_class}")