import numpy as np
import datetime

import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from pandas import read_csv
from sklearn.model_selection import train_test_split
from keras import Sequential
import subprocess
import threading

# Parametry eksperymentu
learning_rates = [0.001, 0.01]
layer_configs = [[64, 32], [128, 64, 32]]
batch_size = [32, 64]

def split_data(X, y):
    X_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Funkcja tworzenia modelu
def create_model(layer_size, learning_rate):
    model = Sequential()
    for size in layer_size:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Funkcja do eksperymentu
def experiment(X_train, y_train, X_val, y_val):
    results = []
    for ler_rat in learning_rates:
        for layers in layer_configs:
            for batch in batch_size:
                print(f"Trening dla learning_rate={ler_rat}, layers={layers}, batch={batch}")
                model = create_model(layers, ler_rat)
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=batch,
                                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
                best_val_accuracy = max(history.history['val_accuracy'])
                results.append({'ler_rat': ler_rat, 'layers': layers, 'batch': batch, 'val_accuracy': best_val_accuracy})

    best_model_params = max(results, key=lambda x: x['val_accuracy'])
    return results, best_model_params

# Ścieżka do danych
wine_path = "./wine.data"

# Kolumny w zbiorze danych
columns = ['Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
           'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
           'Color_intensity', 'Hue', 'OD280/OD315', 'Proline']

# Wczytywanie danych
wine_data = read_csv(wine_path, header=None, names=columns)
data = wine_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Przygotowanie danych
X = data.drop('Class', axis=1).values
y = data['Class'].values.reshape(-1, 1)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Uruchomienie eksperymentu
results, best_params = experiment(X_train, y_train, X_val, y_val)

# Baseline model (model bazowy)
baseline_model = create_model([64, 32], 0.001)
baseline_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)
baseline_accuracy = max(baseline_model.history.history['val_accuracy'])

# Łączenie danych treningowych i walidacyjnych
X_train_val = np.vstack((X_train, X_val))
y_train_val = np.concatenate((y_train, y_val), axis=0)

# Tworzenie modelu na najlepszych parametrach
final_model = create_model(best_params['layers'], best_params['ler_rat'])

# Dodanie callbacku TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Funkcja do uruchomienia TensorBoard
def run_tensorboard():
    subprocess.run(["tensorboard", "--logdir", log_dir])

# Uruchomienie TensorBoard w tle w nowym wątku
tensorboard_thread = threading.Thread(target=run_tensorboard)
tensorboard_thread.start()

# Trening modelu z TensorBoard
final_model.fit(X_train_val, y_train_val, epochs=50, batch_size=best_params['batch'], verbose=0, callbacks=[tensorboard_callback])

# Testowanie na zbiorze testowym
test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)

# Wyświetlanie wyników
print(f"Baseline model accuracy: {baseline_accuracy}")
print(f"Final model accuracy on test set: {test_accuracy}")

results,best_model=experiment(X_train, y_train, X_val, y_val)
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print("Najlepszy model: ",best_model)
