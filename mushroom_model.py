import streamlit as st
import pandas as pd
import pickle


# Load model and label encoders
model = pickle.load(open("dt_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))


# Streamlit UI
st.set_page_config(page_title="Mushroom Classifier", layout="centered")
st.title("🍄 Mushroom Edibility Classifier")

def user_input_features():
    data = {}
    for col, le in label_encoders.items():
        if col == 'class': 
            continue
        # create a title from the column name
        label = col.replace('-', ' ').title()
        data[col] = st.selectbox(label, list(le.classes_))
    return pd.DataFrame([data])

# 1) Get the raw inputs
input_df = user_input_features()

# 2) Encode each column
for col in input_df.columns:
    input_df[col] = label_encoders[col].transform(input_df[col])

# 3) debugging: ensure all expected columns are present
# and fill missing ones with the most frequent class
expected = list(model.feature_names_in_)
for col in expected:
    if col not in input_df.columns:
        # fill with the encoder’s most frequent class
        input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]

# now reorder exactly
input_df = input_df[expected]

# 4) Predict
pred = model.predict(input_df)[0]
label = " ❌ Poisonous" if pred == 1 else "✅ Edible"

# 5) Display
st.subheader("Prediction Result")
st.success(f"The mushroom is predicted to be: **{label}**")
