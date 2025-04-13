import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("saved_bert_model")
model = BertForSequenceClassification.from_pretrained("saved_bert_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Emotion mapping and emojis
emotion_labels = {
    0: ("Sadness", "😢"),
    1: ("Joy", "😊"),
    2: ("Love", "❤️"),
    3: ("Anger", "😠"),
    4: ("Fear", "😨"),
    5: ("Surprise", "😲")
}

# SHAP explainer setup using huggingface_pipeline for compatibility
from transformers import pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=0 if torch.cuda.is_available() else -1)
explainer = shap.Explainer(pipe)

# Streamlit UI
st.title("Text to Emotion Detection with Explainability")
st.write("Enter a sentence and we'll detect the emotion, show the most impactful words (via SHAP), and display an emoji.")

user_input = st.text_area("Enter your sentence:", height=100)
submit = st.button("Detect Emotion")

if submit and user_input.strip() != "":
    # Tokenize and predict
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    label, emoji = emotion_labels[prediction]

    st.subheader("🔍 Prediction")
    st.write(f"**Emotion:** {label} {emoji}")
    st.write("**Confidence:**")
    for idx, prob in enumerate(probs[0]):
        lbl, emj = emotion_labels[idx]
        st.write(f"- {lbl}: {prob.item():.2%} {emj}")

    # SHAP Explanation (bar chart)
    st.subheader("🧠 Most Impactful Words")

    shap_values = explainer([user_input])
    values = shap_values[0].values[:, prediction]
    words = shap_values[0].data

    # Get top 10 impactful words (abs value)
    top_n = 10
    top_indices = np.argsort(np.abs(values))[-top_n:][::-1]
    top_words = [words[i] for i in top_indices]
    top_impacts = [values[i] for i in top_indices]

    # Plotting bar chart
    fig, ax = plt.subplots()
    colors = ["green" if val > 0 else "red" for val in top_impacts]
    ax.barh(top_words[::-1], top_impacts[::-1], color=colors[::-1])
    ax.set_xlabel("Impact on Prediction")
    ax.set_title("Top Impactful Words (SHAP)")
    st.pyplot(fig)

else:
    st.write("👆 Enter a sentence above and click **Detect Emotion** to get started.")


