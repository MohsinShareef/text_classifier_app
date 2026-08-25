import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Text Classifier App",
    page_icon="📂",
    layout="wide"
)

# ============================================
# LOAD MODEL AND VECTORIZER (CACHED)
# ============================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('text_classifier_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'text_classifier_model.pkl' and 'vectorizer.pkl' are in the same directory.")
        return None, None

model, vectorizer = load_artifacts()

# ============================================
# CREATE TABS: PREDICTOR + DOCUMENTATION
# ============================================
tab1, tab2 = st.tabs(["🔮 Predict Text", "📖 Project Documentation"])

# ============================================
# TAB 1: PREDICTOR
# ============================================
with tab1:
    st.header("📝 Text Category Classifier")
    st.write("Enter a sentence or short paragraph below. The model will predict whether it belongs to:")
    col1, col2, col3, col4 = st.columns(4)
    col1.info("**Technology**")
    col2.info("**Politics**")
    col3.info("**Entertainment**")
    col4.info("**Sports**")
    
    st.markdown("---")
    
    user_input = st.text_area("✍️ Input Text", height=150, placeholder="Type or paste your text here...")
    
    if st.button("🚀 Predict", type="primary"):
        if model is None or vectorizer is None:
            st.stop()
        
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text before predicting.")
        else:
            # Transform input
            input_vectorized = vectorizer.transform([user_input])
            
            # Predict
            prediction = model.predict(input_vectorized)[0]
            probabilities = model.predict_proba(input_vectorized)[0]
            classes = model.classes_
            
            # Display result
            st.success(f"**Predicted Category:**  {prediction}")
            
            # Show confidence bar chart
            prob_df = pd.DataFrame({
                'Category': classes,
                'Probability': probabilities
            })
            
            st.subheader("📊 Prediction Confidence")
            st.bar_chart(prob_df.set_index('Category'))
            
            # Show confidence percentage
            confidence = np.max(probabilities) * 100
            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")

# ============================================
# TAB 2: PROJECT DOCUMENTATION
# ============================================
with tab2:
    st.header("📖 Project Documentation")
    st.markdown("---")
    
    # 1. Introduction
    st.subheader("1. Introduction")
    st.write("""
    This project builds a **text classification system** that automatically assigns a given text snippet to one of four predefined categories:  
    **Technology**, **Politics**, **Entertainment**, or **Sports**.  
    
    The system is powered by a **Multinomial Naive Bayes** classifier trained on a labeled dataset of 85 text samples. This interactive web application (built with Streamlit) allows users to get instant predictions.
    """)
    
    # 2. Objective
    st.subheader("2. Project Objectives")
    st.markdown("""
    - **Primary Goal**: Develop a robust, lightweight, and interpretable classifier for multi‑class text categorization.
    - **Secondary Goal**: Deploy the model as an interactive web app to demonstrate real‑world applicability.
    - **Evaluation**: Measure performance using accuracy and a confusion matrix to understand classification strengths and weaknesses.
    """)
    
    # 3. Dataset Overview
    st.subheader("3. Dataset Overview")
    st.write("The dataset (`synthetic_text_data.csv`) contains **85 samples** distributed as follows:")
    
    data_dist = pd.DataFrame({
        "Label": ["Technology", "Politics", "Entertainment", "Sports"],
        "Count": [30, 21, 19, 15]
    })
    st.dataframe(data_dist, hide_index=True, use_container_width=True)
    
    st.caption("Each sample is a short sentence that clearly belongs to one of the four domains (e.g., 'AI in healthcare' → Technology).")
    
    # 4. Why Build This Project?
    st.subheader("4. Why Build This Project?")
    st.markdown("""
    - **Practical Use‑Case**: Automatically classifying news articles, social media posts, or customer feedback into topical buckets helps with content organization, recommendation systems, and trend analysis.
    - **Educational Value**: Demonstrates the entire ML pipeline – from data prep, feature extraction, model training, evaluation, to deployment – using a simple yet effective algorithm.
    - **Demonstration of NLP Basics**: Showcases how raw text is transformed into numerical features (bag‑of‑words) and how a probabilistic model makes predictions.
    - **Deployment Experience**: Creating a Streamlit app provides hands‑on experience with MLOps and user‑friendly interfaces.
    """)
    
    # 5. Why Multinomial Naive Bayes?
    st.subheader("5. Why Multinomial Naive Bayes?")
    st.write("The **Multinomial Naive Bayes (MNB)** classifier is a natural choice for text classification tasks for several reasons:")
    
    advantages = {
        "Simplicity": "MNB is easy to implement, interpret, and debug.",
        "Efficiency": "It trains very quickly and requires minimal computational resources, making it ideal for rapid prototyping.",
        "Effectiveness": "Despite its 'naive' independence assumption, it performs surprisingly well on high‑dimensional sparse data like text, especially when the feature space consists of word counts.",
        "Probabilistic Output": "It provides class probabilities, which can be used to gauge prediction confidence.",
        "Well‑Suited for Count Data": "The multinomial distribution models the frequency of words in a document, which aligns perfectly with the bag‑of‑words representation.",
        "Baseline Model": "MNB serves as an excellent baseline; if its performance is acceptable, more complex models may not be needed."
    }
    for key, value in advantages.items():
        st.markdown(f"**{key}**  \n{value}")
    
    # 6. Model Definition & Theory
    st.subheader("6. Model Definition & Theory (Simplified)")
    st.markdown("""
    **Multinomial Naive Bayes** is a probabilistic classifier based on Bayes’ theorem with the assumption that features (word counts) are conditionally independent given the class.
    
    - **Bayes’ Theorem** for a document *d* and class *c*:  
      `P(c | d) = (P(c) * P(d | c)) / P(d)`  
      Since `P(d)` is constant for all classes, we choose the class that maximises `P(c) * P(d | c)`.
    
    - **Multinomial Likelihood**: For a document represented as a vector of word counts, the model estimates the probability of each word occurring in a document of a given class using **Laplace smoothing** to handle unseen words.
    
    - **Training** involves computing the prior `P(c)` and the conditional probabilities `P(word | c)` from the training data.
    
    - **Prediction** is made by computing the posterior probability for each class and selecting the one with the highest value.
    """)
    
    # 7. Implementation Pipeline
    st.subheader("7. Implementation Pipeline")
    st.markdown("""
    1. **Load Data** – Read the CSV file containing text and labels.
    2. **Split Data** – 80% training, 20% testing.
    3. **Vectorization** – Convert raw text into a Bag-of-Words matrix using `CountVectorizer`.
    4. **Train Model** – Fit a `MultinomialNB` classifier on the vectorized training data.
    5. **Evaluate** – Compute accuracy and generate a confusion matrix heatmap.
    6. **Save Artifacts** – Persist the trained model and vectorizer using `joblib`.
    7. **Deploy** – Load artifacts into this Streamlit app for interactive predictions.
    """)
    
    # 8. Results & Insights
    st.subheader("8. Results & Insights")
    st.markdown("""
    - The model achieves **high accuracy** (typically >80% on this dataset) given the limited sample size.
    - The **confusion matrix** helps identify which categories are easily confused (e.g., Technology vs. Politics if both discuss policy implications of tech).
    - The app displays **prediction confidence** via a bar chart, allowing users to see how certain the model is.
    """)
    # (Optionally, you can paste an actual confusion matrix image here if you saved one, but text description is fine)
    
    # 9. Technologies Used
    st.subheader("9. Technologies Used")
    st.code("""
    - Python 3.x
    - Pandas & NumPy (Data manipulation)
    - Scikit-learn (CountVectorizer, MultinomialNB, metrics)
    - Matplotlib & Seaborn (Visualization)
    - Joblib (Model persistence)
    - Streamlit (Web application framework)
    """, language="text")
    
    # 10. Future Improvements
    st.subheader("10. Future Improvements")
    st.markdown("""
    - Collect a larger, more balanced dataset.
    - Experiment with TF‑IDF vectorization or n‑grams.
    - Try other classifiers (e.g., Logistic Regression, SVM) for comparison.
    - Add confidence thresholds or handle out‑of‑distribution inputs.
    - Deploy on Streamlit Community Cloud or Hugging Face Spaces for public access.
    """)
    
    st.markdown("---")
    st.caption("📌 This documentation accompanies the project code and is intended to provide clear, comprehensive insight into the 'why' and 'how' behind the implementation.")