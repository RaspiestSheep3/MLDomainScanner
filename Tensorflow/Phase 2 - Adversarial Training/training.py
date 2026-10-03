import tensorflow as tf
from tensorflow import keras # pyright: ignore[reportMissingModuleSource] #type : ignore
from tensorflow.keras import layers  # pyright: ignore[reportMissingModuleSource] #type : ignore
#from DomainCalculations import *
import matplotlib.pyplot as plt
import random

#Settings
VOCAB_SIZE = 39
EMBEDDING_DIM = 32
RNN_UNITS = 128

vocab = list("qwertyuiopasdfghjklzxcvbnm1234567890-_") + ["<EOL>"]
indexCharConversion = {i: char for i, char in enumerate(vocab)}
charIndexConversion = {char: i for i, char in enumerate(vocab)}

print(charIndexConversion)

EOL_INDEX = charIndexConversion["<EOL>"]

generator = keras.Sequential([
    layers.Input(shape=(None,)),
    
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    
    layers.LSTM(
        units=RNN_UNITS,
        return_sequences = True,
        recurrent_initializer='glorot_uniform'
    ),
    
    layers.Dense(units=VOCAB_SIZE, activation='softmax')
])

discriminator = keras.Sequential([
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

generator.summary()

def TensorToString(tensor, index_to_char):
    chars = []
    
    for index in tensor.numpy():
        if index == EOL_INDEX:
            break
        chars.append(index_to_char[index])
    
    return "".join(chars)

def GenerateDomain(generator, startingLetter, max_length=70):
    currentInput = tf.constant([[charIndexConversion[startingLetter]]])
    generatedIndices = [charIndexConversion[startingLetter]]
    
    for _ in range(max_length - 1):
       
        predictions = generator(currentInput)
        
        nextCharLogits = predictions[:, -1, :]        
        sampledIndex = tf.random.categorical(nextCharLogits, num_samples=1)[0, 0].numpy()
        
        if sampledIndex == EOL_INDEX:
            break
            
        generatedIndices.append(sampledIndex)
        
        currentInput = tf.concat([currentInput, [[sampledIndex]]], axis=1)
        
    domain = TensorToString(tf.constant(generatedIndices), indexCharConversion)
    return domain, generatedIndices

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


for i in range(20):
    domain, _ = GenerateDomain(generator, random.choice(vocab[:len(vocab) -1]))
    print(domain, len(domain))