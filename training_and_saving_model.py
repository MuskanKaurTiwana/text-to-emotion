
import torch
print(torch.cuda.is_available())  # Should print True
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # Should show GPU name


import zipfile
import os
# Define the path of the ZIP file
zip_path = "tweetsdataset.zip"  # Update this if your file is elsewhere
# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("tweetsdataset")  # Extract to a folder
# List the extracted files to verify
print(os.listdir("tweetsdataset"))


import pandas as pd
# Load training dataset
train_df = pd.read_csv("tweetsdataset/training.csv") #80%
# Load validation dataset
val_df = pd.read_csv("tweetsdataset/validation.csv") #10%
# Load test dataset
test_df = pd.read_csv("tweetsdataset/test.csv") #10%

# Display first few rows
print(train_df.head())
print("Training Data Columns:", train_df.columns)


#checking for issues in dataset
import re


#checking for missing values
print("Missing Values:\n", train_df.isnull().sum())

# Function to check for specific patterns in the text column
def check_text_issues(df, column_name="text"):
    print("Checking for issues in the dataset...\n")

    # Check for special characters (excluding standard punctuation)
    special_chars = df[df[column_name].str.contains(r"[^a-zA-Z0-9\s,.!?']", regex=True)]
    print(f"Rows with special characters (excluding common punctuation): {len(special_chars)}")

    # Check for excessive punctuation
    excessive_punc = df[df[column_name].str.contains(r"[.,!?]{3,}", regex=True)]
    print(f"Rows with excessive punctuation: {len(excessive_punc)}")

    # Check for extra spaces (leading, trailing, or multiple spaces in between)
    extra_spaces = df[df[column_name].str.contains(r"\s{2,}", regex=True)]
    print(f"Rows with extra spaces: {len(extra_spaces)}")

    # Check for URLs
    urls = df[df[column_name].str.contains(r"https?://\S+|www\.\S+", regex=True)]
    print(f"Rows with URLs: {len(urls)}")

    # Check for @usernames
    usernames = df[df[column_name].str.contains(r"@\w+", regex=True)]
    print(f"Rows with @usernames: {len(usernames)}")

    # Check for hashtags
    hashtags = df[df[column_name].str.contains(r"#\w+", regex=True)]
    print(f"Rows with hashtags: {len(hashtags)}")

    print("\nCheck completed.")

# Run the function on the training dataset
check_text_issues(train_df)


import transformers
print(transformers.__version__)  # Should print the installed version


import torch
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Sample check
sample_text = "I am feeling very happy today!"
encoded_sample = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
print(encoded_sample)


# Tokenize the entire dataset at once
train_encodings = tokenizer(
    train_df["text"].tolist(),  # Convert column to list
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

val_encodings = tokenizer(
    val_df["text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

test_encodings = tokenizer(
    test_df["text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Check tokenized sample
print(train_encodings["input_ids"][0])

#data preparation step


import torch
from torch.utils.data import Dataset, DataLoader

# Extract labels from your dataset (assuming the label column is named 'label')
train_labels = torch.tensor(train_df["label"].tolist())
val_labels = torch.tensor(val_df["label"].tolist())

# Define the EmotionDataset class with warning fixes
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}  # Fix warnings
        item['labels'] = self.labels[idx].clone().detach()  # Fix warnings
        return item

# Convert tokenized data into PyTorch datasets
train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

# Define DataLoaders for batching
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Verify the DataLoader output
batch = next(iter(train_dataloader))
print(batch.keys())  # Expected output: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

#loading bert and optimizer setup


import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW  # Use PyTorch's implementation




# Define the number of emotion labels in your dataset
num_labels = len(set(train_labels.tolist()))  # Get unique labels

# Load pre-trained BERT with a classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Print model architecture
print(model)

# Define Loss Function & Optimizer

import torch
from torch.optim import AdamW

#from transformers import AdamW
from torch.nn import CrossEntropyLoss

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function
criterion = CrossEntropyLoss()

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Training parameters
num_epochs = 3  # Change this as needed

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0

    for batch in train_dataloader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()

    # Print average loss for the epoch
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

print("Training completed!")


import os

# Define the directory to save the model
model_dir = "saved_bert_model"
os.makedirs(model_dir, exist_ok=True)  # Create directory if not exists
# Save model and tokenizer
model.save_pretrained("saved_bert_model")
tokenizer.save_pretrained("saved_bert_model")
# Save optimizer state (optional, if you plan to continue training later)
torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pth"))

print(f"Model and tokenizer saved in '{model_dir}'")

# downloading the bert model in machine for future usage

import shutil

# Zip the model directory
shutil.make_archive("saved_bert_model", 'zip', "saved_bert_model")

# Download the zip file
from google.colab import files
files.download("saved_bert_model.zip")

