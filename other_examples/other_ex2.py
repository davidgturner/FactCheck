from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# Prepare the sentence pairs for each class
examples = {
    "entailment": [
        ("A man is playing a guitar.", "The man is a musician."),
        ("A woman is cooking in the kitchen.", "Someone is cooking."),
        ("Children are playing soccer in the park.", "Kids are playing sports outdoors.")
    ],
    "neutral": [
        ("A man is playing a guitar.", "The man loves music."),
        ("A woman is cooking in the kitchen.", "The woman is a chef."),
        ("Children are playing soccer in the park.", "The children are at school.")
    ],
    "contradiction": [
        ("A man is playing a guitar.", "The man is sleeping."),
        ("A woman is cooking in the kitchen.", "No one is in the kitchen."),
        ("Children are playing soccer in the park.", "The children are reading books indoors.")
    ]
}

# Function to get the string label from the predicted class ID
def get_label_from_id(class_id):
    labels = model.config.id2label  # Get the correct label mapping from the model's config
    return labels[class_id]

# Iterate over examples and make predictions
for label, sentence_pairs in examples.items():
    print(f"Examples of {label.capitalize()}:")
    for premise, hypothesis in sentence_pairs:
        # Tokenize the sentences and obtain model inputs
        inputs = tokenizer(premise, hypothesis, return_tensors="pt")

        # Get model predictions
        outputs = model(**inputs)

        # Obtain the predicted class ID and label
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        print("predicted_class_id ", predicted_class_id)
        
        predicted_label = get_label_from_id(predicted_class_id)

        # Print the result
        print(f"Premise: '{premise}'\nHypothesis: '{hypothesis}'\nPredicted: {predicted_label}\n")

    print("-" * 50)
