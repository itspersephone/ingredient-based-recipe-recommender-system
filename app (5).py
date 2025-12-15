import streamlit as st
import pickle
import pandas as pd
from googleapiclient.discovery import build

# YouTube API Key
YOUTUBE_API_KEY = "AIzaSyD9J7o6mXMe7NQucZzq3V3Fqf_NNilMFww"

# -------------- Load Model Files --------------
@st.cache_resource
def load_files():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_matrix.pkl", "rb"))
    model = pickle.load(open("nn_model.pkl", "rb"))
    df = pd.read_pickle("recipes_df.pkl")
    return vectorizer, tfidf, model, df

vectorizer, tfidf, model, df = load_files()

# -------------- YouTube API Function --------------
def get_youtube_video(query):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            q=query + " recipe tutorial",
            part="snippet",
            maxResults=1,
            type="video"
        )
        response = request.execute()
        if response and response.get('items'):
            item = response['items'][0]
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            return video_id, video_title
        return None, None
    except Exception as e:
        st.error(f"Error calling YouTube API: {e}")
        return None, None

# -------------- Streamlit UI --------------
st.set_page_config(page_title="🔥 AI Recipe Recommender", layout="wide")

# Custom CSS for GUI and buttons
st.markdown("""
<style>
body {
    background-color: #FFF8F0;
}
.stButton>button {
    background-color: #FFE5D9;
    color: #333333;
    height: 2em;
    width: 100%;
    font-size: 14px;
    border-radius: 8px;
    border: 1px solid #D9A48F;
    margin-bottom: 2px;
}
.stButton>button:hover {
    background-color: #FFD1B3;
    color: #333333;
}
.stTextInput>div>div>input {
    height: 2.5em;
    font-size: 16px;
    border-radius: 10px;
    padding-left: 10px;
}
.stSelectbox>div>div>div>select {
    height: 2.5em;
    font-size: 16px;
}
h1 {
    color: #FF6F61;
}
h3 {
    color: #FF6F61;
}
h4 {
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

st.title("🔥 AI Recipe Recommender")
st.write("Enter the ingredients you have. Example: `flour, sugar, milk, butter`")

user_input = st.text_input("Ingredients (comma-separated)")
priority = st.selectbox("Prioritize by:", ["None", "Time", "Rating", "Popularity"])

if st.button("Recommend Recipes"):
    if user_input.strip() == "":
        st.error("Please enter ingredients.")
    else:
        user_list = [i.lower().strip() for i in user_input.split(",")]
        user_text = " ".join(user_list)
        user_vec = vectorizer.transform([user_text])
        distances, indices = model.kneighbors(user_vec, n_neighbors=10)

        recommended = []
        for idx in indices[0]:
            recipe = df.iloc[idx]
            recipe_info = recipe.copy()
            recipe_info['similarity'] = 1 - distances[0][list(indices[0]).index(idx)]
            recommended.append(recipe_info)

        if priority == "Time":
            recommended.sort(key=lambda x: x['minutes'])
        elif priority == "Rating":
            recommended.sort(key=lambda x: x['avg_rating'], reverse=True)
        elif priority == "Popularity":
            recommended.sort(key=lambda x: x['popularity_score'], reverse=True)

        st.subheader("🍽️ Recommended Recipes:")
        for rank, recipe in enumerate(recommended[:5]):
            recipe_name = recipe['name'].title()
            st.markdown(f"### {rank+1}. {recipe_name}")

            # Display stats as labeled buttons with smaller size and reduced gap
            col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
            col1.button(f"Rating: ⭐ {recipe['avg_rating']:.2f}", key=f"rating_{rank}")
            col2.button(f"Time: ⏱️ {recipe['minutes']} min", key=f"time_{rank}")
            col3.button(f"Similarity: 🔍 {recipe['similarity']:.2f}", key=f"similarity_{rank}")
            col4.button(f"Popularity: 🔥 {int(recipe['popularity_score'])}", key=f"popularity_{rank}")
            col5.button(f"Total Ratings: 🗳️ {int(recipe['rating_count'])}", key=f"totalratings_{rank}")

            # ------------------- YouTube Video -------------------
            video_id, video_title = get_youtube_video(recipe_name)
            if video_id:
                st.markdown(f"**🎥 Video Tutorial:** [{video_title}](https://www.youtube.com/watch?v={video_id})")
                st.video(f"https://www.youtube.com/watch?v={video_id}", start_time=0, format="video/mp4", width=400)
            else:
                st.markdown("**🎥 Video Tutorial:** No relevant video found.")

            # ------------------- Expanders -------------------
            with st.expander("📝 Nutrition"):
                nut = recipe["nutrition"]
                labels = ["Calories", "Total Fat (%)", "Sugar (%)", "Sodium (%)", "Protein (%)",
                          "Saturated Fat (%)", "Carbohydrates (%)"]
                nutrition_values = eval(nut) if isinstance(nut, str) else nut
                for label, value in zip(labels, nutrition_values):
                    st.markdown(f"- **{label}:** {value}")

            with st.expander("🛒 Ingredients"):
                for ing in recipe["clean_ingredients"]:
                    st.markdown(f"- {ing}")

            with st.expander("👩‍🍳 Steps"):
                try:
                    steps_list = eval(recipe["steps"]) if isinstance(recipe["steps"], str) else recipe["steps"]
                    for step in steps_list:
                        st.markdown(f"- {step}")
                except:
                    st.markdown(recipe["steps"])

            st.markdown("---")
