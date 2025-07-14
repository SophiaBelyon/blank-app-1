import streamlit as st
import pandas as pd
import pickle

# Load model and label encoders
model = pickle.load(open("dt_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Mapping of codes to full descriptive names for each feature
attribute_labels = {
    'cap-shape': {'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 'f': 'Flat', 'k': 'Knobbed', 's': 'Sunken'},
    'cap-surface': {'f': 'Fibrous', 'g': 'Grooves', 'y': 'Scaly', 's': 'Smooth'},
    'cap-color': {'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'r': 'Green', 'p': 'Pink', 'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow'},
    'bruises': {'t': 'Bruises', 'f': 'No'},
    'odor': {'a': 'Almond', 'l': 'Anise', 'c': 'Creosote', 'y': 'Fishy', 'f': 'Foul', 'm': 'Musty', 'n': 'None', 'p': 'Pungent', 's': 'Spicy'},
    'gill-attachment': {'a': 'Attached', 'd': 'Descending', 'f': 'Free', 'n': 'Notched'},
    'gill-spacing': {'c': 'Close', 'w': 'Crowded', 'd': 'Distant'},
    'gill-size': {'b': 'Broad', 'n': 'Narrow'},
    'gill-color': {'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 'g': 'Gray', 'r': 'Green', 'o': 'Orange', 'p': 'Pink', 'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow'},
    'stalk-shape': {'e': 'Enlarging', 't': 'Tapering'},
    'stalk-root': {'b': 'Bulbous', 'c': 'Club', 'u': 'Cup', 'e': 'Equal', 'z': 'Rhizomorphs', 'r': 'Rooted', '?': 'Missing'},
    'stalk-surface-above-ring': {'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'},
    'stalk-surface-below-ring': {'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'},
    'stalk-color-above-ring': {'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'},
    'stalk-color-below-ring': {'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'},
    'veil-type': {'p': 'Partial', 'u': 'Universal'},
    'veil-color': {'n': 'Brown', 'o': 'Orange', 'w': 'White', 'y': 'Yellow'},
    'ring-number': {'n': 'None', 'o': 'One', 't': 'Two'},
    'ring-type': {'c': 'Cobwebby', 'e': 'Evanescent', 'f': 'Flaring', 'l': 'Large', 'n': 'None', 'p': 'Pendant', 's': 'Sheathing', 'z': 'Zone'},
    'spore-print-color': {'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 'r': 'Green', 'o': 'Orange', 'u': 'Purple', 'w': 'White', 'y': 'Yellow'},
    'population': {'a': 'Abundant', 'c': 'Clustered', 'n': 'Numerous', 's': 'Scattered', 'v': 'Several', 'y': 'Solitary'},
    'habitat': {'g': 'Grasses', 'l': 'Leaves', 'm': 'Meadows', 'p': 'Paths', 'u': 'Urban', 'w': 'Waste', 'd': 'Woods'}
}

# Streamlit UI
st.set_page_config(page_title="Mushroom Classifier", layout="centered")
st.title("🍄 Mushroom Edibility Classifier")

# Build user input with full names displayed
def user_input_features():
    data = {}
    for col, le in label_encoders.items():
        if col == 'class': 
            continue
        # Display full descriptive labels for options
        options = [(desc, code) for code, desc in attribute_labels[col].items()]
        label = col.replace('-', ' ').title()
        # format_func shows the descriptive name, but we capture the code for prediction
        selected_code = st.selectbox(label, options, format_func=lambda x: x[0])[1]
        data[col] = selected_code
    return pd.DataFrame([data])

# 1) Get the raw inputs
input_df = user_input_features()

# 2) Encode each column (labels are already codes, so transform directly)
for col in input_df.columns:
    input_df[col] = label_encoders[col].transform(input_df[col])

# 3) Ensure all expected columns are present and filled
expected = list(model.feature_names_in_)
for col in expected:
    if col not in input_df.columns:
        input_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
input_df = input_df[expected]

# 4) Predict
pred = model.predict(input_df)[0]
label = " ❌ Poisonous" if pred == 1 else "✅ Edible"

# 5) Display
st.subheader("Prediction Result")
st.success(f"The mushroom is predicted to be: **{label}**")

