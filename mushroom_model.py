import streamlit as st 
import pandas as pd
import pickle

# mushroom_model.py
import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open("dt_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Mapping dictionaries for user-friendly display
feature_maps = {
    "cap-shape": {'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 'f': 'Flat', 'k': 'Knobbed', 's': 'Sunken'},
    "odor": {'a': 'Almond', 'l': 'Anise', 'c': 'Creosote', 'y': 'Fishy', 'f': 'Foul', 'm': 'Musty', 'n': 'None', 'p': 'Pungent', 's': 'Spicy'},
    "gill-color": {'k': 'Black', 'n': 'Brown', 'g': 'Gray', 'p': 'Pink', 'u': 'Purple', 'w': 'White', 'h': 'Chocolate'},
    "ring-type": {'e': 'Evanescent', 'f': 'Flaring', 'l': 'Large', 'n': 'None', 'p': 'Pendant', 'r': 'Ring', 'z': 'Zone'},
    "stalk-color-above-ring": {'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'}
}

# Inverse map for encoding user selection
inverse_maps = {feature: {v: k for k, v in mapping.items()} for feature, mapping in feature_maps.items()}

# Streamlit UI
st.set_page_config(page_title="Mushroom Classifier", layout="centered")
st.title("🍄 Mushroom Classification App")
st.markdown("Enter mushroom features to check if it's *edible or poisonous*.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    cap_shape = st.selectbox("Cap Shape", list(feature_maps["cap-shape"].values()))
    odor = st.selectbox("Odor", list(feature_maps["odor"].values()))
    gill_color = st.selectbox("Gill Color", list(feature_maps["gill-color"].values()))

with col2:
    ring_type = st.selectbox("Ring Type", list(feature_maps["ring-type"].values()))
    stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", list(feature_maps["stalk-color-above-ring"].values()))

# Predict button
if st.button("Predict"):
    # Convert readable inputs to encoded letters
    input_dict = {
        "cap-shape": inverse_maps["cap-shape"][cap_shape],
        "odor": inverse_maps["odor"][odor],
        "gill-color": inverse_maps["gill-color"][gill_color],
        "ring-type": inverse_maps["ring-type"][ring_type],
        "stalk-color-above-ring": inverse_maps["stalk-color-above-ring"][stalk_color_above_ring]
    }

    # Create dataframe and encode using saved encoders
    input_df = pd.DataFrame([input_dict])
    for col in input_df.columns:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    class_label = label_encoders["class"].inverse_transform([prediction])[0]

    # Display result
    if class_label == 'e':
        st.success("✅ The mushroom is **Edible**.")
    else:
        st.error("⚠️ The mushroom is **Poisonous**.")
