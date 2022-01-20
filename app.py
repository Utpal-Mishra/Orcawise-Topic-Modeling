import home
import two


import streamlit as st

st.audio(open('sweet.mp3', 'rb').read(), format='audio/ogg')

PAGES = {
    "Home": home,
    "Topic Modeling": two,
}

st.sidebar.title('Navigation Bar')

selection = st.sidebar.selectbox("Go to: \n", list(PAGES.keys()))
page = PAGES[selection]
page.app()