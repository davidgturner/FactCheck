import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SummaCEntailment:
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", threshold=0.6):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()
        self.threshold = threshold

    def get_entailment_score(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            entailment_prob = probs[0][2].item()  # Assuming the entailment label index is 2
        return entailment_prob

    def is_sentence_entailed(self, document, summary_sentence):
        doc_sentences = document.split('.')
        max_score = 0
        for sent in doc_sentences:
            score = self.get_entailment_score(sent, summary_sentence)
            if score > max_score:
                max_score = score
        print("max score ", max_score)
        print("threshold ", self.threshold)
        return max_score > self.threshold

    def is_summary_consistent(self, document, summary, consistency_threshold=0.8):
        summary_sentences = summary.split('.')
        entailed_sentences = 0
        total_sentences = 0

        for sent in summary_sentences:
            if sent.strip():  # Ensure the sentence is not empty
                total_sentences += 1
                if self.is_sentence_entailed(document, sent):
                    entailed_sentences += 1

        return (entailed_sentences / total_sentences) >= consistency_threshold

    
    def get_max_entailment_score(self, document, summary_sentence):
        doc_sentences = document.split('.')
        max_score = 0
        for sent in doc_sentences:
            score = self.get_entailment_score(sent, summary_sentence)
            if score > max_score:
                max_score = score
        return max_score

# Example usage:
document = "The cat sat on the mat. Birds fly in the sky. Fish swim in the water."
summary = "Animals sit on mats. Birds can fly."

summac = SummaCEntailment()
summary_sentences = summary.split('.')
for sent in summary_sentences:
    if sent.strip():  # Ensure the sentence is not empty
        score = summac.get_max_entailment_score(document, sent)
        print(f"Entailment score for '{sent}': {score:.2f}")
result = summac.is_summary_consistent(document, summary)
print(f"\nThe summary is {'consistent' if result else 'inconsistent'} with the document.")