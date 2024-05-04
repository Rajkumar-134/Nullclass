import tkinter as tk
from tkinter import ttk
import tensorflow as tf
from keras.layers import TextVectorization
import re
import tensorflow.strings as tf_strings
import json
import string
from keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import threading
import speech_recognition as sr
from googletrans import Translator
from datetime import datetime
import difflib
import tkinter.messagebox as messagebox

# English to Spanish translation
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


# Load the English vectorization layer configuration
with open('eng_vectorization_config.json') as json_file:
    eng_vectorization_config = json.load(json_file)


# Recreate the English vectorization layer with basic configuration
eng_vectorization = TextVectorization(
    max_tokens = eng_vectorization_config['max_tokens'],
    output_mode = eng_vectorization_config['output_mode'],
    output_sequence_length = eng_vectorization_config['output_sequence_length']
)

# Apply the custom standardization function
eng_vectorization.standardize = custom_standardization


# Load the Spanish vectorization layer configuration
with open('spa_vectorization_config.json') as json_file:
    spa_vectorization_config = json.load(json_file)


# Recreate the Spanish vectorization layer with basic configuration
spa_vectorization = TextVectorization(
    max_tokens = spa_vectorization_config['max_tokens'],
    output_mode = spa_vectorization_config['output_mode'],
    output_sequence_length = spa_vectorization_config['output_sequence_length'],
    standardize = custom_standardization
)

# Load and set the English vocabulary
with open('eng_vocab.json') as json_file:
    eng_vocab = json.load(json_file)
    eng_vectorization.set_vocabulary(eng_vocab)
with open('spa_vocab.json') as json_file:
    spa_vocab = json.load(json_file)
    spa_vectorization.set_vocabulary(spa_vocab)

# load the spanish model
transformer = tf.saved_model.load('transformer_model')

spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sentence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = tf.argmax(predictions[0, i, :]).numpy().item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

# load the French model and tokenizers
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

def translate_to_french(english_sentence):
    english_sentence = english_sentence.lower()
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)
    
    english_sentence = english_sentence.reshape((-1,max_length))
    
    french_sentence = model.predict(english_sentence)[0]
    french_sentence = [np.argmax(word) for word in french_sentence]
    french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]
    
    return french_sentence

def translate_to_spanish(english_sentence):
    spanish_sentence = decode_sentence(english_sentence)
    return spanish_sentence.replace("[start]", "").replace("[end]", "")

consecutive_wrong_words = []

def check_word(word, correct_words):
    if word in correct_words:
        return True, None  # Word is correct, no suggestions needed
    else:
        suggestions = difflib.get_close_matches(word, correct_words, n=3)
        return False, suggestions  # Word is incorrect, return suggestions
def suggest_words(event):
    global consecutive_wrong_words
    english_sentence = text_input.get("1.0", "end-1c").lower()

    if event.keysym=='space':
        # Check if the entered English sentence contains incorrect words
        incorrect_words = []
        words = english_sentence.split()
        word = words[-1] if words else ""
        is_correct, suggestions = check_word(word, eng_vocab)
        if not is_correct:
            incorrect_words.append((word, suggestions))
            if len(consecutive_wrong_words) >= 2:
                    # Show error message with list of consecutive wrong words and suggestions
                error_message = "You have entered 2 consecutive wrong words:\n"
                for wrong_word, wrong_suggestions in consecutive_wrong_words:
                    error_message += f"{wrong_word}\n"
                    if wrong_suggestions:
                        error_message += "Suggestions:\n"
                        for suggestion in wrong_suggestions:
                            error_message += f"- {suggestion}\n"
                messagebox.showerror("Consecutive Wrong Words", error_message)
                    # Clear the list of consecutive wrong words after showing the error
                consecutive_wrong_words = []
                return  # Exit the function after showing the error message

                # Append wrong word and suggestions to consecutive wrong words list
            consecutive_wrong_words.append((word, suggestions))
        else:
            consecutive_wrong_words = []  # Reset the list if a correct word is entered
        
        if incorrect_words:
            error_message = "The following words are not available:\n"
            for word, suggestions in incorrect_words:
                error_message += f"{word}\n"
                if suggestions:
                    error_message += "Suggestions:\n"
                    for suggestion in suggestions:
                        error_message += f"- {suggestion}\n"

            # Show error message with suggestions in a messagebox
            messagebox.showerror("Word Suggestions", error_message)
            return  # Exit the function if there are incorrect words

    
def handle_translate():
    global consecutive_wrong_words
    selected_language = language_var.get()
    english_sentence = text_input.get("1.0", "end-1c")
    translation = None  # Initialize the translation variable
        # Translation logic here...
    try:
        if selected_language == "French":
            translation = translate_to_french(english_sentence)
        elif selected_language == "Spanish":
            translation = translate_to_spanish(english_sentence)
    except Exception as e:
        # Handle the translation error gracefully
        translation_output.delete("1.0", "end")
        translation_output.insert("end", f"Error during translation: {e}")

    # Update the translation output
    if translation is not None:
        translation_output.delete("1.0", "end")
        print(translation)
        translation_output.insert("end", f"{selected_language} translation: {translation}")


def is_after_6pm():
    now = datetime.now()
    return now.hour >= 18  # 6 PM in 24-hour format

# Function to handle microphone button click
def start_recording():
    if not is_after_6pm():
        messagebox.showerror("Please try after 6 PM IST.")
        return

    def recognize_speech():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            text_input.delete("1.0", "end")
            translation_output.delete("1.0", "end")
            message_text.delete("1.0", "end") 
            message_text.insert(tk.END, "Listening...\n")
            audio_data = r.record(source, duration=5)  # Record audio for 5 seconds or adjust as needed
        try:
            message_text.insert(tk.END, "Recognizing...\n")
            recognized_text = r.recognize_google(audio_data)
            print("Recognized Text:", recognized_text)
            text_input.delete("1.0", "end")  # Clear existing text
            text_input.insert("end", recognized_text)
            message_text.delete("1.0", "end") 

            # Check if the text starts with "M" or "O"
            if recognized_text and recognized_text[0].upper() in ['M', 'O']:
                raise ValueError("Please repeat, avoiding words starting with M or O.")

            # Translate the recognized English text to Hindi
            translator = Translator()
            translation = translator.translate(recognized_text, src='en', dest='hi')
            translated_text = translation.text
            translation_output.delete("1.0", "end")  # Clear existing translation
            translation_output.insert("end", translated_text)
        except sr.UnknownValueError:
            messagebox.showerror("Recognition could not understand the audio. Please speak one more time.")
        except ValueError as ve:
            print(f"Error: {ve}")
            translation_output.delete("1.0", "end")  # Clear existing translation
            translation_output.insert("end", str(ve))

    threading.Thread(target=recognize_speech).start()

# Setting up the main window
root = tk.Tk()
root.title("Language Translator")
root.geometry("550x600")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the text to be translated", font=(font_style, font_size, 'bold'))
input_heading.pack()
# Text input for English sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))
text_input.pack()

# Language selection
language_var = tk.StringVar()
language_label = tk.Label(root, text="Select the language to translate to", font=(font_style, font_size, 'bold'))
language_label.pack()
language_select = ttk.Combobox(root, textvariable=language_var, values=["French", "Spanish"], font=(font_style, font_size), state="readonly")
language_select.pack()

# Submit button
submit_button = ttk.Button(root, text="Translate", command=handle_translate)
submit_button.pack(pady=10)

# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)
# Heading for output
output_heading = tk.Label(output_frame, text="Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=5, width=50, font=(font_style, font_size))
translation_output.pack()

# Microphone button
mic_button = ttk.Button(root, text="🎤", command=start_recording)
mic_button.pack(pady=10)
text_input.bind("<KeyPress>", suggest_words)
message_text = tk.Text(root, height=2, width=50, font=(font_style, font_size))
message_text.pack()

# Running the application
root.mainloop()
