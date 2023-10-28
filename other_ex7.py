from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FactChecker:

    def __init__(self, model, tokenizer, threshold=0.7):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.threshold = threshold

    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespaces
        text = ' '.join(text.split())
        # Replace multiple punctuation marks with a single one (e.g., "!!!" becomes "!")
        text = text.replace('...', '.').replace('!!!', '!').replace('??', '?')
        # Optionally, you can add more cleaning steps as needed
        return text

    def get_entailment_scores(self, premises, hypothesis):
        inputs = self.tokenizer(premises, [hypothesis for _ in premises], return_tensors="pt", truncation=True, padding='longest', max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            entailment_label_id = self.model.config.label2id["entailment"]
            entailment_probs = probs[:, entailment_label_id].tolist()
        return entailment_probs

    def is_sentence_entailed(self, doc_sentences, summary_sentence):
        scores = self.get_entailment_scores(doc_sentences, summary_sentence)
        max_score = max(scores)
        return max_score > self.threshold

    def is_summary_consistent(self, doc_sentences, summary_sentences, consistency_threshold=0.7):
        entailed_sentences = sum([self.is_sentence_entailed(doc_sentences, sent) for sent in summary_sentences if sent.strip()])
        return (entailed_sentences / len(summary_sentences)) >= consistency_threshold

    def is_fact_supported_by_passages(self, passages, fact_statements):
        # Clean and pre-split passages into sentences
        all_passage_sentences = [self.clean_text(passage_dict['text']).split('.') for passage_dict in passages]
        
        cleaned_fact_statements = [self.clean_text(stmt) for stmt in fact_statements]
        
        for stmt in cleaned_fact_statements:
            supported = False
            for passage_sentences in all_passage_sentences:
                if self.is_sentence_entailed(passage_sentences, stmt):
                    supported = True
                    break
            if not supported:
                return False
        return True
    


model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)

checker = FactChecker(roberta_ent_model, ent_tokenizer, 0.6)


# Original data
passages = [
    {"title": "Cats", "text": "The cat sat on the mat."},
    {"title": "Birds", "text": "Birds fly in the sky."},
    {"title": "Fish", "text": "Fish swim in the water."}
]
fact = "Animals sit on mats. Birds can fly."

# Modified data
modified_passages = [
    {"title": "Cats", "text": "The CAT sat on the mat!!!"},
    {"title": "Birds", "text": " Birds  fly in the sky..."},
    {"title": "Fish", "text": "Fish swim in the water."}
]
modified_fact = " Animals sit on  mats. Birds can fly."

# Run the entailment checks
result_original = checker.is_fact_supported_by_passages(passages, fact.split('. '))
result_modified = checker.is_fact_supported_by_passages(modified_passages, modified_fact.split('. '))

# Print the results
print(f"Original data: The fact is {'supported' if result_original else 'not supported'} by the passages.")
print(f"Modified data: The fact is {'supported' if result_modified else 'not supported'} by the passages.")
