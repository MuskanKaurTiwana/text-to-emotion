import os
import zipfile
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Added SHAP library
import shap

# Unzip dataset if not already extracted
zip_path = "tweetsdataSet.zip"  # Ensure the correct file name
if zipfile.is_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tweetsdataset")
    print("Dataset extracted successfully!")
else:
    print("Error: tweetsdataset.zip not found or invalid!")

# Unzip saved model
model_zip_path = "saved_bert_model.zip"  # Ensure correct filename
if zipfile.is_zipfile(model_zip_path):
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall("saved_bert_model")
    print("Model extracted successfully!")
else:
    print("Error: saved_bert_model.zip not found or invalid!")


# Load tokenizer and model from saved directory
tokenizer = BertTokenizer.from_pretrained("saved_bert_model")
model = BertForSequenceClassification.from_pretrained("saved_bert_model")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model and tokenizer loaded successfully!")

# Load datasets
train_df = pd.read_csv("tweetsdataset/training.csv")
val_df = pd.read_csv("tweetsdataset/validation.csv")
test_df = pd.read_csv("tweetsdataset/test.csv")

# Tokenization function
def tokenize_data(df):
    return tokenizer(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

# Tokenize datasets
val_encodings = tokenize_data(val_df)
test_encodings = tokenize_data(test_df)

# Convert labels to tensors
val_labels = torch.tensor(val_df["label"].tolist())
test_labels = torch.tensor(test_df["label"].tolist())

# Define PyTorch dataset class
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx].clone().detach()
        return item

# Create PyTorch datasets
val_dataset = EmotionDataset(val_encodings, val_labels)
test_dataset = EmotionDataset(test_encodings, test_labels)

# Define DataLoaders
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Validation and test datasets loaded successfully!")

# #Step 5: Run Validation & Compute Accuracy
# # ==============================

from torch.nn import CrossEntropyLoss

# Define loss function
criterion = CrossEntropyLoss()

# Evaluate function
def evaluate_model(dataloader, dataset_name="Validation"):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)

            # Compute loss
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            probs = softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Compute average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    print(f"{dataset_name} Loss: {avg_loss:.4f}")
    print(f"{dataset_name} Accuracy: {accuracy:.4%}")  # Display as percentage

# Run evaluation
evaluate_model(val_dataloader, "Validation")
evaluate_model(test_dataloader, "Test")








