import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🧠",
    layout="wide"
)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:white;
}

.subtitle{
    text-align:center;
    color:gray;
}

.stTextArea textarea{
    background-color:#1e293b;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🧠 AI Fake Job Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect fraudulent job postings using Machine Learning</p>', unsafe_allow_html=True)

st.divider()

# Layout columns
col1, col2 = st.columns([2,1])

with col1:

    st.subheader("📄 Enter Job Description")

    text = st.text_area(
        "Paste Job Description Here",
        height=200,
        placeholder="Paste job posting..."
    )

    if st.button("🔍 Predict Job Authenticity"):

        if text.strip() == "":
            st.warning("Please enter job description")

        else:

            data = vectorizer.transform([text])
            prediction = model.predict(data)
            prob = model.predict_proba(data)[0][1]

            st.divider()

            if prediction[0] == 1:
                st.error("⚠️ Fake Job Posting Detected")
            else:
                st.success("✅ Real Job Posting")

            st.metric(
                label="Fraud Probability",
                value=str(round(prob*100,2)) + "%"
            )

with col2:

    st.subheader("📊 System Info")

    st.info("""
    **AI Model:** Logistic Regression  
    **Technique:** TF-IDF NLP  
    **Dataset:** Fake Job Postings  
    """)

    st.subheader("🚨 Fraud Indicators")

    st.write("""
    - Unrealistic salary
    - No experience required
    - Asking for payment
    - Requesting bank details
    - Immediate hiring
    """)

st.divider()

st.caption("AI Fake Job Detection System • Built with Streamlit")
