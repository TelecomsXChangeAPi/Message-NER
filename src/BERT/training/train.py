import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

def preprocess_tags(df):
    # Map categorical tags to integers based on your update
    tag_mapping = {"Email": 0, "URL": 1}
    # Update 'Entity' to 'Tag' if your tag information is in the 'Tag' column
    df['Tag'] = df['Tag'].map(tag_mapping)  # Directly map tags using the updated column
    return df

class NERDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_len):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]  # Assuming each text is already a single sentence.
        tag = self.tags[item]

        encoding = self.tokenizer.encode_plus(
                                  text,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  padding='max_length',
                                  truncation=True,
                                  return_attention_mask=True,
                                  return_tensors='pt')

        # Create a label list which assigns the tag to all tokens of the input
        labels = [tag] * encoding['input_ids'].size(1)
        labels = np.array(labels)
        labels[0] = -100  # Set label for [CLS] token to -100
        labels[-1] = -100  # Set label for [SEP] token to -100
        labels[encoding['input_ids'].size(1) - encoding['attention_mask'].sum():] = -100  # Set labels for padding tokens to -100

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = NERDataset(
        texts=df['Sentence'].to_numpy(),
        tags=df['Tag'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def main():
    # Load and preprocess Dataset
    df = pd.read_csv('ner-dataset.csv')  # Update this path
    df = preprocess_tags(df)

    BATCH_SIZE = 16
    MAX_LEN = 128
    EPOCHS = 3

    train_df, test_df = train_test_split(df, test_size=0.1)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Adjust num_labels to the number of unique tags
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Check for MPS device availability and use it if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
   
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_loss = train_epoch(model, train_data_loader, optimizer, device)
        print(f'Train Loss: {train_loss}')

    torch.save(model.state_dict(), 'ner_model.pth')

if __name__ == '__main__':
    main()
