import json
import sqlite3
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras #type : ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def SaveModel(extraCode = ""):
    if(extraCode) != "":
        extraCode = "-" + extraCode
    model.save(f"model{datetime.datetime.now().strftime('%d.%m.%y-%H.%M.%S')}{extraCode}.keras")

with open("PythonSettings.json", "r") as f:
    settings = json.load(f)
print(settings)

conn = sqlite3.connect(settings["DB Path"])
cursor = conn.cursor()

model = keras.Sequential([
    keras.layers.Dense(6, activation='tanh'), 
    keras.layers.Dense(12, activation='leaky_relu'), 
    keras.layers.Dense(32, activation='leaky_relu'), 
    keras.layers.Dense(6, activation='leaky_relu'), 
    keras.layers.Dense(1, activation='sigmoid'), 
])

learningRate = settings["Learning Rate"]
model.compile(
    optimizer=Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learningRate, decay_steps=settings['Num Training Entries'] // 16, decay_rate=0.98, staircase=True)), 
    loss="binary_crossentropy" 
)

cursor.execute(f"SELECT * FROM domainsTesting LIMIT {settings['Num Training Entries']}")
rows = cursor.fetchall()

inputs = []
outputs = []
for row in rows:
    inputs.append(np.array(row[1:7], dtype=float).flatten())
    outputs.append(row[-1])
    
inputs = np.array(inputs, dtype=float)
outputs = np.array(outputs, dtype=float)

model.fit(inputs, outputs, epochs=settings["Num Epochs"], verbose=2, batch_size=16)
SaveModel()