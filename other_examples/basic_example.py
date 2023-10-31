from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# Prepare the sentences
premise = "A man is playing a guitar."
hypothesis = "The man is a musician."

# Tokenize the sentences and obtain model inputs
inputs = tokenizer(premise, hypothesis, return_tensors="pt")

# Get model predictions
outputs = model(**inputs)

# Obtain the predicted class
predicted_class = torch.argmax(outputs.logits, dim=1).item()

# Print the result
print(f"Predicted class: {predicted_class}")
