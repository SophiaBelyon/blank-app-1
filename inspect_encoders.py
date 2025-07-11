import pickle
enc = pickle.load(open("label_encoders.pkl", "rb"))
print("Number of encoders:", len(enc))
print("Keys:", enc.keys())
