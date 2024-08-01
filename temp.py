import streamlit as st
from PIL import Image
import pickle
import os
import numpy as np
import streamlit.components.v1 as components

with open ("crop_model.pkl",'rb') as file:
    crop_model = pickle.load(file)



st.title("Agrise")
st.header("Crop Prediction")
location_js = """
<script>
navigator.geolocation.getCurrentPosition(
    (position) => {
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;
        document.getElementById("location-data").value = `${latitude},${longitude}`;
        document.getElementById("location-form").dispatchEvent(new Event("submit"));
    }
);
</script>
"""
location_html = """
<form id="location-form">
    <input type="hidden" id="location-data" name="location-data" />
</form>
"""

components.html(location_js + location_html)

location = st.query_params.get("location-data", [""])[0]

if location:
    st.write(f"Location: {location}")



uploaded_file = st.file_uploader("Upload a picture of the crop", type=["jpg", "jpeg", "png"])
dataset_folder="crop_images"

class_labels={class_name:label for label,class_name in enumerate(os.listdir(dataset_folder))}

if uploaded_file is not None:
    # Display the uploaded picture
    def predict_crop_type(uploaded_file,crop_model,class_labels):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Crop Image", use_column_width=True)
        image = image.resize((224, 224))
        image_array=np.array(image)
        image_array=image_array/255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions=crop_model.predict(image_array)
        predicted_class_idx=np.argmax(predictions)
        predicted_class=[class_name for class_name,class_idx in class_labels.items() if class_idx==predicted_class_idx]
        predicted_class=predicted_class[0
        ]
        print(predicted_class)
        return predicted_class

    

    predicted_crop = predict_crop_type(uploaded_file,crop_model,class_labels)
    st.write(f"Predicted Crop: {predicted_crop}")
    print(class_labels)

