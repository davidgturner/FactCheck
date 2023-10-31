from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# Single fact
fact = "Children are playing soccer in the park."

# Group of passages
passages = [
    "Kids are playing sports outdoors.",
    "The children are at school.",
    "The children are reading books indoors.",
    "A group of kids is playing football in the garden.",
    "Children are running around in the park, some are playing with a ball."
]

# Function to get the string label from the predicted class ID
def get_label_from_id(class_id):
    labels = model.config.id2label
    return labels[class_id]

# Evaluate the fact against the group of passages
support_count = 0
total_passages = len(passages)

for passage in passages:
    # Tokenize the sentences and obtain model inputs
    inputs = tokenizer(fact, passage, return_tensors="pt")

    # Get model predictions
    outputs = model(**inputs)

    # Obtain the predicted class ID and label
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = get_label_from_id(predicted_class_id)

    # Print the result
    print(f"Fact: '{fact}'\nPassage: '{passage}'\nPrediction: {predicted_label}\n")
    
    # Count the number of supports (entailments)
    if predicted_label == "entailment":
        support_count += 1

# Print the final binary assessment based on majority vote
if support_count > total_passages / 2 or (support_count == total_passages / 2 and total_passages % 2 == 0):
    print("Final Assessment: Supported")
else:
    print("Final Assessment: Not Supported")
