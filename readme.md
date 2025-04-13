# 🧠 Text-to-Emotion Detection with Explainability (BERT + SHAP)

This project is a BERT-based emotion classification system that not only detects emotions from input text but also explains which words influenced the prediction using SHAP (SHapley Additive exPlanations). The frontend is built with **Streamlit**, making it interactive and user-friendly.

## 🔍 Features

- Emotion detection using a fine-tuned BERT model.
- SHAP-based word importance visualization for explainability.
- Interactive UI using Streamlit.
- Dataset split into training, validation, and test sets.

## 📁 Project Structure

.
├── app.py                      # Streamlit app for UI and explainability
├── training_and_saving_model.py     # Code to train the BERT model and save it
├── validation_and_testing.py  # Code to validate/test the model and compute accuracy
├── requirements.txt           # All required Python libraries
├── tweetsdataset/              # Folder containing training, validation, and test datasets
│   ├── training.csv
│   ├── validation.csv
│   └── test.csv
└── saved_bert_model/          # Folder with the trained model and tokenizer

## ⚙️ How to Run

1. **Clone this repository:**

2. **Install dependencies:**

pip install -r requirements.txt

3. **Train the model (optional, model already saved):**

python training_and_saving.py

4. **Run validation and testing:**

python validation_and_testing.py


5. **Launch the Streamlit app:**

streamlit run app.py

## 📊 Emotions Detected

- 😢 Sadness  
- 😊 Joy  
- ❤️ Love  
- 😠 Anger  
- 😨 Fear  
- 😲 Surprise  

## 🧠 Explainability with SHAP

SHAP highlights the most impactful words in the text that contributed to the emotion prediction, offering transparency and trust in the model’s decision.

## 💡 Future Scope

- Extend the model with more training data.
- Add multilingual support.
- Improve explainability using attention-based visualization.
- Predicting emotions for negative sentences(including not)
