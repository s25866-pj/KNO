import StandardScaler
import pandas as pd
import numpy as np
from keras_tuner import Hyperband
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import Hyperband
import os.path

# Pobierz dane
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
file_patch="wine.data"
columns = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]
if os.path.isfile(file_patch):
    data = pd.read_csv(file_patch, header=None, names=columns)
else:
    data = pd.read_csv(url, header=None, names=columns)
"""
    url: Adres URL do zestawu danych Wine z repozytorium UCI.
    columns: Lista nazw kolumn dla danych. Każda kolumna odpowiada jednej cesze (lub etykiecie Class).
    pd.read_csv: Wczytanie danych CSV z podanego URL-a i przypisanie nazw kolumn.
"""
X = data.drop("Class", axis=1).values
y = data["Class"].values
"""
    X: Macierz cech (dane wejściowe). Usunięcie kolumny Class, która jest etykietą.
    y: Wektor etykiet klas.
"""
# Standaryzacja cech
scaler = StandardScaler()
X = scaler.fit_transform(X)
"""
    StandardScaler: Przekształca każdą cechę do skali o średniej 0 i odchyleniu standardowym 1.
    fit_transform: Dopasowanie i transformacja danych.
"""
# Kodowanie etykiet
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
"""
    LabelEncoder: Przekształca etykiety klas na liczby całkowite.
"""

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
    train_test_split: Losowo dzieli dane na zbiory:
    Treningowy: 80%.
    Testowy: 20%.
    random_state=42: Gwarantuje powtarzalność podziału.

"""
class WineModel(keras.Model):
    def __init__(self, num_classes, hidden_units, dropout_rate):
        super(WineModel, self).__init__()
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(layers.Dense(units, activation='relu'))
            self.hidden_layers.append(layers.Dropout(dropout_rate))
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def build_model(self, input_shape):
        self.build(input_shape)
"""
    WineModel: Klasa definiująca sieć neuronową dziedziczącą po keras.Model.
    __init__: Konstruktor przyjmujący:
    num_classes: Liczba klas wyjściowych.
    hidden_units: Liczba jednostek w ukrytych warstwach.
    dropout_rate: Współczynnik regularizacji dropout.
    hidden_layers: Lista warstw ukrytych (Dense i Dropout).
    output_layer: Warstwa wyjściowa z aktywacją softmax.
    call: Definiuje przepływ danych przez model.
    build_model: Buduje model dla zadanego kształtu danych wejściowych.
"""
num_classes=len(np.unique(y))
model=WineModel(num_classes=num_classes,hidden_units=[64,64,64],dropout_rate=0.2)


def build_model(hp):
    hidden_units = [
        hp.Int(f"units_layer_{i}", min_value=32, max_value=128, step=16) for i in range(2)
    ]
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)/10

    model = WineModel(num_classes=num_classes, hidden_units=hidden_units, dropout_rate=dropout)
    model.build_model((None, X_train.shape[1]))  # Build the model with input shape
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", [1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
"""
hp.Int i hp.Float: Określenie przestrzeni poszukiwań dla hiperparametrów (liczby jednostek w warstwach i dropout).
hp.Choice: Wybór spośród zadanych wartości (np. learning_rate).
Model jest budowany z dopasowanymi hiperparametrami.
Adnotacje 1e-3 oznacza adnotację naukową jako 1 do potęgi -3 ->(0.001)
"""
tuner=Hyperband(
    build_model,
    objective="accuracy",
    max_epochs=20,
    factor=3,
    directory="wine_tuning",
    project_name="wine_tuning",
)
"""
Hyperband: Algorytm optymalizujący hiperparametry:
build_model: Funkcja do budowy modeli.
objective: Metryka optymalizowana (tu: accuracy).
max_epochs: Maksymalna liczba epok trenowania.
factor: Współczynnik zmniejszania liczby prób w kolejnych etapach.
"""
tuner.search(X_train, y_train,validation_split=0.2,epochs=20,callbacks=[keras.callbacks.EarlyStopping(patience=5)])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
"""
tuner.search: Przeprowadzenie procesu wyszukiwania na danych treningowych.
EarlyStopping: Zatrzymanie trenowania, jeśli metryka przestaje się poprawiać przez 5 epok.
"""
model=tuner.hypermodel.build(best_hps)
model.fit(X_train,y_train,validation_split=0.2,epochs=20,callbacks=[keras.callbacks.EarlyStopping(patience=5)])
test_loss, test_acc = model.evaluate(X_test,y_test,verbose=0)
print(f"doładność na zbiorze testowym: {test_acc:.2f}")