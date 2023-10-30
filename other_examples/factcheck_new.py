import re
import string
import numpy as np
import torch
import nltk
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download necessary NLTK data only once
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


class FactExample:
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return f"fact={repr(self.fact)}; label={repr(self.label)}; passages={repr(self.passages)}"


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = {}  # Cache for storing results

    def classify_entailment(self, probabilities):
        max_class = max(probabilities, key=probabilities.get)
        if max_class == "neutral":
            max_class = 'entailment' if probabilities['entailment'] > probabilities['contradiction'] else 'contradiction'
        return max_class

    def get_label_from_id(self, class_id):
        return self.model.config.id2label[class_id]

    def check_entailment(self, premise: str, hypothesis: str):
        cache_key = (premise, hypothesis)
        if cache_key in self.cache:
            return self.cache[cache_key]

        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Obtain the predicted class ID and label
            predicted_class_id = torch.argmax(logits, dim=1).item()
            predicted_label = self.get_label_from_id(predicted_class_id)
                
            # Calculate the confidence of the prediction
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence = probabilities[0][predicted_class_id].item()

        # predicted_class_id = torch.argmax(logits, dim=1).item()
        # predicted_label = self.get_label_from_id(predicted_class_id)

        # probabilities = torch.nn.functional.softmax(logits, dim=1)

        # # Extracting the confidence for the predicted class
        # predicted_probabilities = probabilities[0]
        # confidence = predicted_probabilities[predicted_class_id].item()

        self.cache[cache_key] = (predicted_label, confidence)
        return predicted_label, confidence

class WordRecallThresholdFactChecker(object):

    def cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_a = np.linalg.norm(vector1)
        norm_b = np.linalg.norm(vector2)
        similarity = dot_product / (norm_a * norm_b)
        return similarity
   
    def jaccard_similarity(str1: str, str2: str) -> float:
        """Compute the Jaccard similarity between two strings."""
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def overlap_coefficient(self, tokenized_facts: str, tokenized_passage: str):
        numerator = len(set(tokenized_facts) & set(tokenized_passage))
        denominator = min(len(set(tokenized_facts)), len(set(tokenized_passage)))
        overlap_coefficient = numerator / denominator
        return overlap_coefficient
    
    def predict(self, fact: str, passages: List[dict]) -> str:
        # threshold Checking
        threshold: float = 0.60
        average_similarity = self.evaluate_similarity(fact, passages)
        meets_threshold = average_similarity >= threshold
        prediction = ""
        if meets_threshold:
            prediction = "S"
        else:
            prediction = "NS"

        return prediction

    def evaluate_similarity(self, fact, passages) -> float:
        tokenized_facts = nltk.word_tokenize(fact)

        stem: bool = False # True
        remove_punctuation : bool = True
        remove_stop_words: bool = False
        similarity_metric: str = 'overlap'
        
        stemmer = nltk.stem.PorterStemmer()

        if remove_punctuation:
            tokenized_facts = [word for word in tokenized_facts if word not in string.punctuation]

        if stem:
            tokenized_facts = [stemmer.stem(word) for word in tokenized_facts]
        
        stop_words = set(nltk.corpus.stopwords.words('english'))

        if remove_stop_words:
            tokenized_facts = [word for word in tokenized_facts if word.lower() not in stop_words]

        # Remove Empty Strings
        tokenized_facts = [word for word in tokenized_facts if word]

        results = []
        for passage in passages:
            passage_text = passage['title'] + ' ' + passage['text']
            tokenized_passage = nltk.word_tokenize(passage_text)

            tokenized_passage = [word for word in tokenized_passage if word]

            if remove_punctuation:
                tokenized_passage = [word for word in tokenized_passage if word not in string.punctuation]

            if stem:
                tokenized_passage = [stemmer.stem(word) for word in tokenized_passage]
        
            if remove_stop_words:
                tokenized_passage = [word for word in tokenized_passage if word.lower() not in stop_words]
        
            # word frequency vectors
            word_set = set(tokenized_facts + tokenized_passage)
            vector1 = [tokenized_facts.count(word) for word in word_set]
            vector2 = [tokenized_passage.count(word) for word in word_set]

            # similarity calculation
            if similarity_metric == 'cosine':
                sim = self.cosine_similarity(vector1, vector2)
            elif similarity_metric == 'jaccard':
                sim = self.jaccard_similarity(vector1, vector2)
            elif similarity_metric == 'overlap':
                sim = self.overlap_coefficient(tokenized_facts, tokenized_passage)
            else:
                raise ValueError(f'Unsupported similarity metric: {similarity_metric}')
        
            # add similarity result to results list
            results.append(sim)

        # compute average similarity score
        average_similarity = np.average(results) if results else 0
        return average_similarity

# class MyFactChecker:
#     def __init__(self, ent_model):
#         self.ent_model = ent_model

#     @staticmethod
#     def clean_text(text: str) -> str:
#         text = text.lower()
#         text = re.sub(r'\s+', ' ', text)
#         text = text.replace("</s>", "").replace("<s>", "")
#         text = ''.join(char for char in text if ord(char) < 128)  # fix unicode
#         return text

#     def is_sentence_entailed(self, fact, passage, support_threshold):
#         label, score = self.ent_model.check_entailment(fact, passage)
#         return label == "entailment" and score > support_threshold

#     def is_fact_supported_by_passages(self, fact, passages, support_threshold):
#         # all_passage_sentences = [self.clean_text(passage_dict['text']).split('.') for passage_dict in passages]
#         all_passage_sentences = [nltk.sent_tokenize(self.clean_text(passage_dict['title'] + "." + passage_dict['text'])) for passage_dict in passages]
#         supported = any(self.is_sentence_entailed(fact, passage_sentence, support_threshold) for passage_sentences in all_passage_sentences for passage_sentence in passage_sentences)
#         return supported

class MyFactChecker:
    def __init__(self, ent_model):
        self.ent_model = ent_model

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.replace("</s>", "").replace("<s>", "")
        text = ''.join({char for char in text if ord(char) < 128})  # fix unicode
        return text

    def is_sentence_entailed(self, fact, sentence, support_threshold):
        label, score = self.ent_model.check_entailment(fact, sentence)
        return label == "entailment" and score > support_threshold

    def is_fact_supported_by_passages(self, fact, passages, support_threshold) -> str:
        # First, tokenize the passages into sentences
        all_passage_sentences = [nltk.sent_tokenize(passage_dict['text']) for passage_dict in passages]
        
        # Flatten the list of lists into a single list of sentences
        all_passage_sentences = [sentence for sublist in all_passage_sentences for sentence in sublist]

        # Now, check if each cleaned sentence supports the fact
        for passage_sentence in all_passage_sentences:
            cleaned_sentence = self.clean_text(passage_sentence)
            if self.is_sentence_entailed(fact, cleaned_sentence, support_threshold):
                return "S"
        return "NS"

class EntailmentFactChecker:
    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.word_recall_fact_checker = WordRecallThresholdFactChecker()
        self.checker = MyFactChecker(self.ent_model)

    # @staticmethod
    # def clean_text(text):
    #     text = text.lower()
    #     text = re.sub(r'\s+', ' ', text)
    #     # text = text.replace("</s>", "").replace("<s>", "")
    #     text = ''.join(char for char in text if ord(char) < 128)  # fix unicode
    #     text = re.sub(r'<s>', ' ', text)  # Replace start tags with space
    #     text = re.sub(r'</s>', ' ', text)  # Replace end tags with space
    #     text = re.sub(r'[^a-zA-Z\s]', '', text).strip()  # Remove non-alphabetic characters
    #     text = re.sub(r'[^\w\s]', '', text)
    #     return text
    
    @staticmethod
    def clean_text(text):
        text = text.lower()
        # text = ''.join(char for char in text if ord(char) < 128)  # fix unicode
        text = re.sub(r'<s>', ' ', text)  # Replace start tags with space
        text = re.sub(r'</s>', ' ', text)  # Replace end tags with space
        text = re.sub(r'[^a-zA-Z\s]', '', text).strip()  # Remove non-alphabetic characters
        # text = re.sub(r'\s+', ' ', text)  # Reduce multiple spaces to a single space
        return text

    def get_avg_conf_support(self, fact, passages, support_threshold) -> str:
        support_confidences = []
        contradiction_confidences = []

        # First, tokenize the passages into sentences
        all_passage_sentences = [nltk.sent_tokenize(passage_dict['text']) for passage_dict in passages]
        
        # Flatten the list of lists into a single list of sentences
        all_passage_sentences = [sentence for sublist in all_passage_sentences for sentence in sublist]

        # Now, clean each sentence and check for entailment
        for passage_sentence in all_passage_sentences:
            cleaned_sentence = self.clean_text(passage_sentence)
            predicted_label, confidence = self.ent_model.check_entailment(fact, cleaned_sentence)
            if predicted_label == "entailment":
                support_confidences.append(confidence)
            elif predicted_label == "contradiction":
                contradiction_confidences.append(confidence)

        avg_support_confidence = sum(support_confidences) / len(support_confidences) if len(support_confidences) >= 0 else 0

        return "S" if avg_support_confidence > support_threshold else "NS"

    def predict(self, fact: str, passages: List[dict]) -> str:
        cleaned_fact = self.clean_text(fact)

        # first use the overlap prediction model, if it doesn't pass that then throw it out
        LOW_CONF_THRESHOLD = 0.50
        similarity_conf = self.word_recall_fact_checker.evaluate_similarity(fact, passages)
        if similarity_conf < LOW_CONF_THRESHOLD:
            return "NS"
        
        a : str = self.get_final_assessment_method(cleaned_fact, passages)
        return a
        #b : str = self.get_avg_conf_support(cleaned_fact, passages, support_threshold=0.50)
        #return b
        # c : str = self.checker.is_fact_supported_by_passages(cleaned_fact, passages, support_threshold=0.50)
        # return c
        # if (a == b == c):
        #     return a
        # else:
        #     return "NS"

    def get_final_assessment_method(self, cleaned_fact, passages):
        sentence_predictions = []

        # First, tokenize the passages into sentences
        all_passage_sentences = [nltk.sent_tokenize(passage["text"]) for passage in passages]
        
        # Flatten the list of lists into a single list of sentences
        all_passage_sentences = [sentence for sublist in all_passage_sentences for sentence in sublist]

        # Now, clean each sentence and check for entailment
        for sentence in all_passage_sentences:
            cleaned_sentence = self.clean_text(sentence)
            print("clean fact ", cleaned_fact, " clean sentence ", cleaned_sentence)
            prediction = self.ent_model.check_entailment(cleaned_fact, cleaned_sentence)[0]  # index 0 is the label
            print(" prediction ", prediction)
            sentence_predictions.append(prediction)

        final_prediction_counts = {
            "entailment": sentence_predictions.count("entailment"),
            "neutral": sentence_predictions.count("neutral"),
            "contradiction": sentence_predictions.count("contradiction")
        }
        final_assessment = max(final_prediction_counts, key=lambda k: (final_prediction_counts[k], k == "entailment"))

        return "S" if final_assessment == "entailment" or (final_assessment == "neutral" and final_prediction_counts["entailment"] >= final_prediction_counts["contradiction"]) else "NS"
