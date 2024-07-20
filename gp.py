import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os

def load_siamese_model(model_path):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        st.error("Model file not found.")
        return None

def test_siamese_model(siamese_model, img1, dataset_images):
    input_shape = (128, 128, 3)
    img1 = img_to_array(img1.resize(input_shape[:2]))
    img1 = img1 / 255.0
    img1 = np.expand_dims(img1, axis=0)

    similarities = []
    for img2 in dataset_images:
        img2_resized = img2.resize(input_shape[:2])
        img2_array = img_to_array(img2_resized) / 255.0
        img2_array = np.expand_dims(img2_array, axis=0)
        similarity = siamese_model.predict([img1, img2_array])
        similarities.append(similarity[0][0])

    
    most_similar_idx = np.argmax(similarities)
    most_similar_img = dataset_images[most_similar_idx]

    return most_similar_img, most_similar_idx


def main():
    st.title('Simulation de la géolocalisation des drones par vision par ordinateur.')
    input_shape = (128, 128, 3)  
    
    uploaded_file = st.file_uploader("Upload Query Image", type=['jpg', 'jpeg', 'png'])
    

    if uploaded_file is not None :
        model_path = 'E:/PFE/SNN1.h5'  
        dataset_path = 'E:/validation' 
        df_path='E:/image_coordinates_vald.csv'
        siamese_model = load_siamese_model(model_path)
        if siamese_model is not None:
            img1 = Image.open(uploaded_file)

           
            dataset_images = [Image.open(os.path.join(dataset_path, img_name)) for img_name in os.listdir(dataset_path)]

            
            st.image(img1, caption='Query Image', use_column_width=True)

            
            most_similar_img, most_similar_idx = test_siamese_model(siamese_model, img1, dataset_images)

           
            df = pd.read_csv(df_path)
            if 'Image Name' in df.columns and 'Longitude' in df.columns and 'Latitude' in df.columns:
                image_name = df.iloc[most_similar_idx]['Image Name']
                longitude = df.iloc[most_similar_idx]['Longitude']
                latitude = df.iloc[most_similar_idx]['Latitude']
                st.success("Most Similar Image:")
                st.image(most_similar_img, caption=f'Most Similar Image: {image_name}', use_column_width=True)
                st.title(f"Image Name: {image_name}")
                st.title(f"Longitude: {longitude}")
                st.title(f"Latitude: {latitude}")
            else:
                st.error("CSV file should contain 'image_name', 'longitude', and 'latitude' columns.")
    else:
        st.warning("Please upload the query image.")

if __name__ == '__main__':
    import tensorflow as tf


    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main()
