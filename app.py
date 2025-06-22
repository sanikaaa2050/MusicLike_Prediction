# Step 1: Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 2: Set Page Config
st.set_page_config(page_title="🎵 Music Like Prediction", layout="centered")

# Step 3: Load Dataset
try:
    df = pd.read_csv("your_music_dataset.csv")
    df.columns = df.columns.str.strip().str.lower()
    if 'liked' in df.columns:
        df.rename(columns={'liked': 'like'}, inplace=True)
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# Step 4: Encode categorical columns using fresh encoders
gender_encoder = LabelEncoder()
genre_encoder = LabelEncoder()
music_encoder = LabelEncoder()

try:
    df['gender'] = gender_encoder.fit_transform(df['gender'])
    df['genre_preference'] = genre_encoder.fit_transform(df['genre_preference'])
    df['music_name'] = music_encoder.fit_transform(df['music_name'])
except Exception as e:
    st.error(f"❌ Encoding error: {e}")
    st.stop()

# Step 5: Prepare input options from encoders
gender_options = list(gender_encoder.classes_)
genre_options = list(genre_encoder.classes_)
music_options = list(music_encoder.classes_)

# Step 6: Sidebar Input
with st.sidebar:
    st.header("📌 Enter Listener Info")
    age = st.slider("Age", 10, 70, 25)
    gender = st.selectbox("Gender", gender_options)
    genre = st.selectbox("Genre Preference", genre_options)
    music = st.selectbox("Music Name", music_options)
    minutes = st.slider("Minutes Listened", 0, 300, 30)

# Step 7: Encode user input safely
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0

input_data = pd.DataFrame([{
    'age': age,
    'gender': safe_encode(gender_encoder, gender),
    'genre_preference': safe_encode(genre_encoder, genre),
    'music_name': safe_encode(music_encoder, music),
    'minutes_listened': minutes
}])

# Step 8: Model Training
# Drop user_id if it exists
if 'user_id' in df.columns:
    df = df.drop("user_id", axis=1)

X = df.drop("like", axis=1)
y = df["like"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Step 9: UI Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧑‍🎤 Input", "🎯 Prediction", "📊 Visualization", "🗂️ Dataset"])

# Tab 1: Input Summary
with tab1:
    st.header("🧾 Your Input Summary")
    st.table(input_data)

# Tab 2: Prediction
with tab2:
    st.header("🎧 Will the user like this song?")
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("👍 Yes! The user is likely to like the song.")
    else:
        st.warning("👎 No. The user is unlikely to like the song.")
    st.metric("📈 Model Accuracy", f"{accuracy:.2%}")

# Tab 3: Visualization
with tab3:
    st.header("🎨 Minutes Listened vs Like")
    try:
        fig, ax = plt.subplots()
        sns.boxplot(x=df["like"], y=df["minutes_listened"], ax=ax)
        ax.set_xlabel("Like (0 = No, 1 = Yes)")
        ax.set_ylabel("Minutes Listened")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Could not generate plot: {e}")

# Tab 4: Dataset Preview
with tab4:
    st.header("📂 First 5 Rows")
    st.dataframe(df.head())
    with st.expander("Show All Columns"):
        st.write(df.columns.tolist())
