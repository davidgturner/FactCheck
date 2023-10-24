import re
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Function to clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# Single fact
fact = "Children are playing soccer in the park."

# Group of passages
passages = [
    "Kids are playing sports outdoors. They seem to be having a lot of fun.",
    "The children are at school. They are attending their classes.",
    "The children are reading books indoors. It's a quiet and peaceful environment.",
    "A group of kids is playing football in the garden. They are very energetic.",
    "Children are running around in the park, some are playing with a ball. It's a sunny day."
]

# Function to get the string label from the predicted class ID
def get_label_from_id(class_id):
    labels = model.config.id2label
    return labels[class_id]

# Clean the fact
fact = clean_text(fact)

# Evaluate the fact against the group of passages
support_count = 0
total_passages = len(passages)

for passage in passages:
    # Tokenize the passage into sentences
    sentences = sent_tokenize(passage)
    
    max_prediction = "contradiction"  # Initialize with the minimum value
    
    for sentence in sentences:
        # Clean the sentence
        sentence = clean_text(sentence)
        
        # Tokenize the sentences and obtain model inputs
        inputs = tokenizer(fact, sentence, return_tensors="pt")

        # Get model predictions
        outputs = model(**inputs)

        # Obtain the predicted class ID and label
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = get_label_from_id(predicted_class_id)
        
        # Update the max prediction if the current sentence entails the fact
        if predicted_label == "entailment":
            max_prediction = "entailment"
            break  # No need to check further if we already found an entailment

    # Print the result
    print(f"Fact: '{fact}'\nPassage: '{passage}'\nMax Prediction: {max_prediction.capitalize()}\n")
    
    # Count the number of supports (entailments)
    if max_prediction == "entailment":
        support_count += 1

# Print the final binary assessment based on majority vote
if support_count > total_passages / 2 or (support_count == total_passages / 2 and total_passages % 2 == 0):
    print("Final Assessment: Supported")
else:
    print("Final Assessment: Not Supported")
