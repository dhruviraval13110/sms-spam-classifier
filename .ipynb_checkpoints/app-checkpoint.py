import streamlit as st
import joblib
import re


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("spam_calibrated_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

/* Main title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 17px;
    margin-bottom: 30px;
}

/* Result cards */
.spam-card {
    background: #fee2e2;
    border: 1px solid #fecaca;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-top: 20px;
}

.ham-card {
    background: #dcfce7;
    border: 1px solid #bbf7d0;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 26px;
    font-weight: 700;
}

.confidence {
    font-size: 32px;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
}

.info-card {
    background: #f8fafc;
    border-radius: 14px;
    padding: 18px;
    border: 1px solid #e5e7eb;
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">📱 SMS Spam Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered SMS classification using NLP & Machine Learning</div>',
    unsafe_allow_html=True
)


# =========================================================
# MESSAGE INPUT
# =========================================================

message = st.text_area(
    "💬 Enter your SMS message",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)


# =========================================================
# EXAMPLE MESSAGES
# =========================================================

st.markdown("### 🧪 Try an example")

col1, col2 = st.columns(2)

with col1:
    spam_example = st.button(
        "🚨 Spam Example",
        use_container_width=True
    )

with col2:
    ham_example = st.button(
        "✅ Normal Example",
        use_container_width=True
    )

if spam_example:
    message = "Congratulations! You have won a free iPhone! Click here to claim your prize now!"

if ham_example:
    message = "Hey, are we meeting at college tomorrow?"


# =========================================================
# CLASSIFY BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Message",
    use_container_width=True,
    type="primary"
):

    if not message.strip():

        st.warning("⚠️ Please enter an SMS message first.")

    else:

        # Clean message
        cleaned_message = clean_text(message)

        # Convert to TF-IDF
        message_tfidf = tfidf.transform([cleaned_message])

        # Prediction
        prediction = model.predict(message_tfidf)[0]

        # Probabilities
        probabilities = model.predict_proba(message_tfidf)[0]

        confidence = max(probabilities) * 100


        # =================================================
        # RESULT
        # =================================================

        if prediction == 1:

            st.markdown(
                f"""
                <div class="spam-card">
                    <div class="result-title">🚨 SPAM MESSAGE</div>
                    <p>This message is likely to be spam.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="ham-card">
                    <div class="result-title">✅ NOT SPAM</div>
                    <p>This message appears to be legitimate.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # CONFIDENCE
        # =================================================

        st.markdown(
            '<p style="text-align:center;">Model Confidence</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">{confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.progress(confidence / 100)


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("🧠 About this model"):

    st.write(
        """
        This application uses Natural Language Processing (NLP)
        and Machine Learning to classify SMS messages as spam or
        legitimate.

        **Pipeline**

        • Text Cleaning  
        • TF-IDF Vectorization  
        • Calibrated Linear SVM  
        • Spam / Ham Classification  
        • Confidence Estimation
        """
    )

    st.write(
        "**Best model:** Linear SVM"
    )

    st.write(
        "**Test accuracy:** 98.07%"
    )

    st.write(
        "**Spam F1-score:** 0.92"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Built with Python • Scikit-learn • NLP • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)