import streamlit as st
from transformers import pipeline


st.set_page_config(page_title="Language Detection App", page_icon="🌐")

@st.cache_resource
def load_model():
    return pipeline('text-classification', model='papluca/xlm-roberta-base-language-detection')

language_detector = load_model()

# Mapping of language codes to full names
LANGUAGE_MAP = {
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ur': 'Urdu',
    'fa': 'Persian',
    'tr': 'Turkish',
    # Add more mappings as needed
}

def get_language_full_name(code):
    return LANGUAGE_MAP.get(code, code)

# Streamlit app layout
st.title("🌐 Language Detection App")
st.write("""
This app detects the language of the given text using a pre-trained model from Hugging Face.
Enter your text below and click **Detect Language** to see the results.
""")

# Text input
text = st.text_area("Enter text:", "Type your text here...", height=200)

# Buttons
col1, col2 = st.columns(2)

with col1:
    detect_button = st.button("Detect Language")

with col2:
    clear_button = st.button("Clear Text")

# Clear text action
if clear_button:
    st.experimental_rerun()

# Detect language action
if detect_button:
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Detecting language..."):
            result = language_detector(text)
            label, score = result[0]['label'], result[0]['score']
            full_name = get_language_full_name(label)
            st.success("Detection Complete!")
            st.markdown(f"### Detected Language: **{full_name}**")
            st.markdown(f"### Confidence Score: **{score:.4f}**")

# Footer
st.markdown("---")
st.markdown("Made by [Subhi Arjaria](https://github.com/subhiarjaria18)")
