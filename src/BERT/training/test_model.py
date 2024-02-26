import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# Define the class labels your model was trained on
# Ensure these are in the same order as during training
label_map = {0: "Email", 1: "URL"}

def load_model(model_path, num_labels):
    # Load the trained model and tokenizer
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return model, tokenizer

def predict(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # Predict
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
    
    # Process the predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # Convert predictions to labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [label_map[label] if label in label_map else "O" for label in predictions[0].numpy()]
    
    # Print out tokens with their predicted labels
    for token, label in zip(tokens, predicted_labels):
        print(f"{token} -> {label}")

# Define the path to your saved model and the number of labels
model_path = 'ner_model.pth'
num_labels = 2  # Update this if you have more than two labels

# Load the model and tokenizer
model, tokenizer = load_model(model_path, num_labels)

# Your test sentence
text = "This is dia jamous, my die mail is diajamous@Yahoo.com and my website is https://www.dia.com"

# Predict
predict(text, model, tokenizer)
