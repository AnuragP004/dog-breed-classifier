import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the trained model
MODEL_PATH = './mnet_dog_breed_model.h5'  # Replace with your model filename
model = load_model(MODEL_PATH)



# Set the image size (based on your model's input)
IMG_SIZE = (256, 256)

# List of 70 dog breeds (replace with your actual class names)
dog_breeds = [
    'Afghan', 'African Wild Dog', 'Airedale', 'American Hairless', 'American Spaniel',
    'Basenji', 'Basset', 'Beagle', 'Bearded Collie', 'Bermaise', 'Bichon Frise',
    'Blenheim', 'Bloodhound', 'Bluetick', 'Border Collie', 'Borzoi', 'Boston Terrier',
    'Boxer', 'Bull Mastiff', 'Bull Terrier', 'Bulldog', 'Cairn', 'Chihuahua',
    'Chinese Crested', 'Chow', 'Clumber', 'Cockapoo', 'Cocker', 'Collie', 'Corgi',
    'Coyote', 'Dalmation', 'Dhole', 'Dingo', 'Doberman', 'Elk Hound', 'French Bulldog',
    'German Sheperd', 'Golden Retriever', 'Great Dane', 'Great Perenees', 'Greyhound',
    'Groenendael', 'Irish Spaniel', 'Irish Wolfhound', 'Japanese Spaniel', 'Komondor',
    'Labradoodle', 'Labrador', 'Lhasa', 'Malinois', 'Maltese', 'Mex Hairless',
    'Newfoundland', 'Pekinese', 'Pit Bull', 'Pomeranian', 'Poodle', 'Pug',
    'Rhodesian', 'Rottweiler', 'Saint Bernard', 'Schnauzer', 'Scotch Terrier',
    'Shar_Pei', 'Shiba Inu', 'Shih-Tzu', 'Siberian Husky', 'Vizsla', 'Yorkie'
]

# App title
st.title("🐶 Dog Breed Classifier")

with st.expander("ℹ️ Click to see supported dog breeds (70 total)"):
    st.markdown("""
    ```
    Afghan, African Wild Dog, Airedale, American Hairless, American Spaniel,
    Basenji, Basset, Beagle, Bearded Collie, Bermaise, Bichon Frise,
    Blenheim, Bloodhound, Bluetick, Border Collie, Borzoi, Boston Terrier,
    Boxer, Bull Mastiff, Bull Terrier, Bulldog, Cairn, Chihuahua,
    Chinese Crested, Chow, Clumber, Cockapoo, Cocker, Collie, Corgi,
    Coyote, Dalmation, Dhole, Dingo, Doberman, Elk Hound, French Bulldog,
    German Sheperd, Golden Retriever, Great Dane, Great Perenees, Greyhound,
    Groenendael, Irish Spaniel, Irish Wolfhound, Japanese Spaniel, Komondor,
    Labradoodle, Labrador, Lhasa, Malinois, Maltese, Mex Hairless,
    Newfoundland, Pekinese, Pit Bull, Pomeranian, Poodle, Pug,
    Rhodesian, Rottweiler, Saint Bernard, Schnauzer, Scotch Terrier,
    Shar_Pei, Shiba Inu, Shih-Tzu, Siberian Husky, Vizsla, Yorkie
    ```
    """)


# Upload image
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = dog_breeds[np.argmax(preds)]
    


    st.subheader(f"Prediction: {predicted_class}")
