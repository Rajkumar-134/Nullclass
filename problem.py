import numpy as np
from keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

model = load_model('english_to_french_model')
with open('english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)
    
with open('french_tokenizer.json') as f:
    data = json.load(f)
    french_tokenizer = tokenizer_from_json(data)

with open('sequence_length.json') as f:
    max_length = json.load(f)
    
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')
# Example input sequence with start token
start_token = "[start]"

input_sequence = np.array([[start_token]])

# Predict the next word probabilities
prediction = model.predict(input_sequence)
predicted_word_index = np.argmax(prediction[0])

# Get the predicted word based on the index
predicted_word = tokenizer.index_word[predicted_word_index]

print("Predicted Word:", predicted_word)
