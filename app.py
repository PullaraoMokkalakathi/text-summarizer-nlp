import streamlit as st
import nltk
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data (important for deployment)
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit Page Config
st.set_page_config(page_title="NLP Text Summarizer", page_icon="📝")

# Title
st.title("📝 Text Summarizer using NLP")
st.write("This tool extracts the most important sentences from your text using NLP techniques.")

# Input Box
text_input = st.text_area("Paste your long paragraph here:", height=250)

# Button
if st.button("Summarize Now"):

    if not text_input.strip():
        st.warning("Please paste some text first!")

    else:
        sentences = sent_tokenize(text_input)

        # If text is already short
        if len(sentences) <= 3:
            st.info("Text is already short!")
            st.write(text_input)

        else:
            # Preprocessing
            stop_words = set(stopwords.words("english"))
            words = word_tokenize(text_input.lower())

            # Word Frequency
            freq_table = {}
            for word in words:
                if word.isalnum() and word not in stop_words:
                    freq_table[word] = freq_table.get(word, 0) + 1

            # Sentence Scoring
            sentence_scores = {}
            for sent in sentences:
                for word in freq_table:
                    if word in sent.lower():
                        sentence_scores[sent] = sentence_scores.get(sent, 0) + freq_table[word]

            # Select Top 3 Sentences
            summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
            summary = " ".join(summary_sentences)

            # Output
            st.success("✅ Summary Generated!")
            st.subheader("📄 Summary:")
            st.write(summary)

            # Word Reduction Info
            original_len = len(text_input.split())
            summary_len = len(summary.split())

            st.info(f"Reduced from {original_len} words to {summary_len} words.")