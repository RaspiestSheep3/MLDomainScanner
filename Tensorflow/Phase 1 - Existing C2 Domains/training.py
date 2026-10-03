import json
import sqlite3
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras # pyright: ignore[reportMissingModuleSource] #type : ignore
from plyer import notification
from tensorflow.keras.optimizers import AdamW # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore

def SaveModel(extraCode = ""):
    if(extraCode) != "":
        extraCode = "-" + extraCode
    model.save(f"model{datetime.datetime.now().strftime('%d.%m.%y-%H.%M.%S')}{extraCode}.keras")

    notification.notify(
        title="Model Saved",
        message=f"Successfully saved model{datetime.datetime.now().strftime('%d.%m.%y-%H.%M.%S')}{extraCode}.keras",
        timeout=10
    )

with open("PythonSettings.json", "r") as f:
    settings = json.load(f)
print(settings)

conn = sqlite3.connect(settings["DB Path"])
cursor = conn.cursor()

oldModelPath = input("Load Model (leave blank to not load) : ")

if(oldModelPath.strip() != ""):
    model = keras.models.load_model(oldModelPath)
    model.summary()
else:
    model = keras.Sequential([
        keras.layers.Input(shape=(11,)),
        
        keras.layers.Dense(32, activation='swish'),
        keras.layers.BatchNormalization(),
        #keras.layers.Dropout(0.2),
        
        #keras.layers.Dense(32, activation='leaky_relu'),
        #keras.layers.Dropout(0.2),
        
        keras.layers.Dense(64, activation='swish'),
        keras.layers.BatchNormalization(),
                
        keras.layers.Dense(32, activation='swish'),
        keras.layers.BatchNormalization(),
        
        keras.layers.Dense(16, activation='swish'),
        
        keras.layers.Dense(1, activation='sigmoid')
    ])

learningRate = settings["Learning Rate"]

learningRateReducer = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=30, 
    min_lr=1e-10
)

model.compile(
    optimizer=AdamW(learningRate, weight_decay = 1e-3), 
    loss="binary_crossentropy",
    metrics=["accuracy"] 
)

cursor.execute(f"SELECT * FROM domainsTesting LIMIT {settings['Num Training Entries']}")
trainRows = cursor.fetchall()
trainInputs = np.array([row[1:12] for row in trainRows], dtype=float)
trainOutputs = np.array([row[-1] for row in trainRows], dtype=float)

cursor.execute("SELECT * FROM domainsValidation")
valRows = cursor.fetchall()
valInputs = np.array([row[1:12] for row in valRows], dtype=float)
valOutputs = np.array([row[-1] for row in valRows], dtype=float)

stopper = EarlyStopping(
    monitor='val_loss',         
    patience=200,              
    restore_best_weights=True  
)

model.fit(
    trainInputs, 
    trainOutputs, 
    epochs=settings["Num Epochs"], 
    verbose=2, 
    batch_size=settings["Batch Size"], 
    validation_data=(valInputs, valOutputs),
    callbacks=[stopper, learningRateReducer]  
)

SaveModel()