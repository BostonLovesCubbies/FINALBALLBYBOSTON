import streamlit as st
from card import build_chart

st.set_page_config(page_title="MLB Pitcher Cards", page_icon="🏟", layout="wide")
st.title("🏟 MLB Pitcher Cards")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    pitcher_id = st.text_input("MLB Pitcher ID", placeholder="e.g. 605288")
with col2:
    year = st.selectbox("Season", [2025, 2024, 2023, 2022])
with col3:
    st.write(""); st.write("")
    go = st.button("Generate Card", type="primary", use_container_width=True)

st.caption("Find pitcher IDs at baseballsavant.mlb.com — search a pitcher and grab the number from the URL.")

if go and pitcher_id:
    try:
        with st.spinner("Fetching data and building card... (30–60 seconds)"):
            buf, name, yr = build_chart(int(pitcher_id.strip()), year)
        st.success(f"Card generated for {name} ({yr})")
        st.image(buf, use_column_width=True)
        st.download_button("⬇ Download PNG", data=buf,
                           file_name=f"{name.replace(' ','_')}_{yr}_arsenal.png",
                           mime="image/png")
    except Exception as e:
        st.error(f"Error: {e}")
elif go:
    st.warning("Please enter a pitcher ID.")
