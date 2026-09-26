import json
import sqlite3
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras #type : ignore

modelPath = input("Model Path : ")

model = keras.models.load_model(modelPath)

model.summary()

with open("PythonSettings.json", "r") as f:
    settings = json.load(f)

conn = sqlite3.connect(settings["DB Path"])
cursor = conn.cursor()

cursor.execute("SELECT * FROM domainsValidation")
rows = cursor.fetchall()

val_inputs = np.array([row[1:7] for row in rows], dtype=float)
val_outputs = np.array([row[-1] for row in rows], dtype=float)

predictions = model.predict(val_inputs, batch_size=256).flatten()

mae = np.mean(np.abs(predictions - val_outputs))
print(f"Mean Absolute Error: {mae:.4f}")

outputs = val_outputs.tolist()
predictions = predictions.tolist()

truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0

positive = 0
negative = 0

for i in range(len(outputs)):
    if(outputs[i] == 0 and predictions[i] <= 0.5):
        truePositive += 1
    elif(outputs[i] == 0 and predictions[i] > 0.5):
        falsePositive += 1
    elif(outputs[i] == 1 and predictions[i] <= 0.5):
        falseNegative += 1
    else:
        trueNegative += 1
    
    if(outputs[i] == 0):
        positive += 1
    else:
        negative += 1
    
print(f"TP : {truePositive} ({(truePositive * 100/positive):.2f}% of actual positives)")
print(f"FP : {falsePositive} ({(falsePositive * 100/negative):.2f}% of actual negatives)")
print(f"FN : {falseNegative} ({(falseNegative * 100/positive):.2f}% of actual positives)")
print(f"TN : {trueNegative} ({(trueNegative * 100/negative):.2f}% of actual negatives)")