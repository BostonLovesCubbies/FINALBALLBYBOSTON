import streamlit as st

st.set_page_config(page_title="Baseball Analytics", page_icon="⚾", layout="wide")

st.title("⚾ Baseball Analytics")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏟 MLB Pitcher Cards")
    st.markdown("Savant-style arsenal cards for any MLB pitcher. Enter a player ID to pull live Statcast data.")

with col2:
    st.markdown("### 🏫 HS Player Cards")
    st.markdown("Upload your league CSV and generate percentile cards for any high school player in the pool.")

with col3:
    st.markdown("### 📡 TrackMan Cards")
    st.markdown("Upload a TrackMan CSV export and generate a full arsenal card for any pitcher in the file.")

st.markdown("---")
st.caption("Use the sidebar to navigate between tools.")
