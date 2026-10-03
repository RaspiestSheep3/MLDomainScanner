import json
import sqlite3
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras # pyright: ignore[reportMissingModuleSource] #type : ignore

THRESHOLD = 0.554300
modelPath = input("Model Path : ")
modelPath2 = input("2nd Model Path : ")
if(modelPath2.strip() != ""):
    modelPath3 = input("3rd Model Path : ")
else:
    modelPath3 = ""

model = keras.models.load_model(modelPath)
model.summary()

if(modelPath2 != "" and modelPath3 != ""):
    model2 = keras.models.load_model(modelPath2)
    model3 = keras.models.load_model(modelPath3)

with open("PythonSettings.json", "r") as f:
    settings = json.load(f)

conn = sqlite3.connect(settings["DB Path"])
cursor = conn.cursor()

cursor.execute("""
    SELECT 
        shannonEntropy, vowelConsonantRatio, longestConsecutiveConsonants, 
        dictionaryCount, bigramCount, trigramCount, totalLength, 
        digitPercentage, longestConsecutiveDigits, letterDigitSymbolTransitionCount, 
        indexOfCoincidence, 
        COUNT(DISTINCT isMalicious) AS distinct_labels,
        COUNT(*) AS total_occurrences,
        GROUP_CONCAT(domain || ' (label:' || isMalicious || ')', ' | ') AS overlapping_domains
    FROM domainsTesting
    GROUP BY 
        shannonEntropy, vowelConsonantRatio, longestConsecutiveConsonants, 
        dictionaryCount, bigramCount, trigramCount, totalLength, 
        digitPercentage, longestConsecutiveDigits, letterDigitSymbolTransitionCount, 
        indexOfCoincidence
    HAVING distinct_labels > 1
""")

collisions = cursor.fetchall()

print(f"Total conflicting feature sets within training: {len(collisions)}\n")

cursor.execute("""
    SELECT 
        shannonEntropy, vowelConsonantRatio, longestConsecutiveConsonants, 
        dictionaryCount, bigramCount, trigramCount, totalLength, 
        digitPercentage, longestConsecutiveDigits, letterDigitSymbolTransitionCount, 
        indexOfCoincidence, 
        COUNT(DISTINCT isMalicious) AS distinct_labels,
        COUNT(*) AS total_occurrences,
        GROUP_CONCAT(domain || ' (label:' || isMalicious || ')', ' | ') AS overlapping_domains
    FROM domainsValidation
    GROUP BY 
        shannonEntropy, vowelConsonantRatio, longestConsecutiveConsonants, 
        dictionaryCount, bigramCount, trigramCount, totalLength, 
        digitPercentage, longestConsecutiveDigits, letterDigitSymbolTransitionCount, 
        indexOfCoincidence
    HAVING distinct_labels > 1
""")

collisions = cursor.fetchall()

print(f"Total conflicting feature sets within validation: {len(collisions)}\n")

"""for row in collisions:
    features = row[:11]
    occurrences = row[12]
    domains = row[13]
    
    print(f"Occurrences: {occurrences}")
    print(f"Features   : {features}")
    print(f"Domains    : {domains}")
    print("-" * 80)"""
    
cursor.execute("SELECT * FROM domainsTesting")
rows = cursor.fetchall()

print(f"Length of training : {len(rows)}")

cursor.execute("SELECT * FROM domainsValidation")
rows = cursor.fetchall()

print(f"Length of validation : {len(rows)}")

val_inputs = np.array([row[1:12] for row in rows], dtype=float)
val_outputs = np.array([row[-1] for row in rows], dtype=float)

if(modelPath2 == ""):
    predictions = model.predict(val_inputs, batch_size=256).flatten()
else:
    p1 = model.predict(val_inputs, batch_size=256).flatten()
    p2 = model2.predict(val_inputs, batch_size=256).flatten()
    p3 = model3.predict(val_inputs, batch_size=256).flatten()
    
    predictions = (p1 + p2 + p3) / 3.0

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
    if(outputs[i] == 0 and predictions[i] <= THRESHOLD):
        truePositive += 1
    elif(outputs[i] == 0 and predictions[i] > THRESHOLD):
        falsePositive += 1
    elif(outputs[i] == 1 and predictions[i] <= THRESHOLD):
        falseNegative += 1
    else:
        trueNegative += 1
    
    if(outputs[i] == 0):
        positive += 1
    else:
        negative += 1

best_acc = 0.0
best_thresh = 0.5

# Test thresholds between 0.40 and 0.60
for threshold in np.arange(0.40, 0.60, 0.0001):
    # Adjust prediction mask based on current threshold
    correct = np.sum((predictions <= threshold) == (val_outputs == 0))
    acc = correct / len(val_outputs)
    
    if acc > best_acc:
        best_acc = acc
        best_thresh = threshold

print(f"Optimal Threshold: {best_thresh:6f}")
print(f"Optimized Accuracy: {best_acc * 100:.5f}%")

print(f"TP : {truePositive} ({(truePositive * 100/positive):.2f}% of actual positives)")
print(f"FP : {falsePositive} ({(falsePositive * 100/negative):.2f}% of actual negatives)")
print(f"FN : {falseNegative} ({(falseNegative * 100/positive):.2f}% of actual positives)")
print(f"TN : {trueNegative} ({(trueNegative * 100/negative):.2f}% of actual negatives)")