import streamlit as st 
import pandas as pd
import pickle

# Load pre-trained model
model = pickle.load(open("dt_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))  # Encoders for categorical inputs

st.set_page_config(page_title="Mushroom Classifier", layout="centered")

# App Title
st.title("🍄 Mushroom Classification App")
st.markdown("Enter mushroom features to check if it's *edible or poisonous*.")

# Feature Inputs
col1, col2 = st.columns(2)

with col1:
    cap_shape = st.selectbox("Cap Shape", ['b', 'c', 'x', 'f', 'k', 's'])  # Add description as needed
    odor = st.selectbox("Odor", ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'])
    gill_color = st.selectbox("Gill Color", ['k', 'n', 'g', 'p', 'u', 'w', 'h'])

with col2:
    ring_type = st.selectbox("Ring Type", ['e', 'f', 'l', 'n', 'p', 'r', 'z'])
    stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", ['n', 'b', 'c', 'g', 'p', 'e', 'w', 'y'])

# Prediction
if st.button("Predict"):
    # Prepare input
    input_dict = {
        "cap-shape": cap_shape,
        "odor": odor,
        "gill-color": gill_color,
        "ring-type": ring_type,
        "stalk-color-above-ring": stalk_color_above_ring
    }

    # Encode inputs using saved LabelEncoders
    input_df = pd.DataFrame([input_dict])
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]

    # Show Result
    if prediction == 'e':
        st.success("The mushroom is *Edible*.")
    else:
        st.error("The mushroom is *Poisonous*!")