# factcheck.py

import re
import string
import torch
import numpy as np
import spacy
import gc

import nltk
from nltk.corpus import stopwords
from typing import List, Dict, Union
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('stopwords')
nltk.download('punkt')

class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # print("ENTAILMENT MODEL TYPE THIS DERIVES FROM ", type(self.model))
        # self.

    def classify_entailment(self, probabilities, threshold=0.5):
        probabilities = {
                    'entailment': probabilities['entailment'],
                    'neutral': probabilities['neutral'],
                    'contradiction': probabilities['contradiction']
                }
        max_class = max(probabilities, key=probabilities.get)
        if max_class == "neutral":
            if probabilities['entailment'] > probabilities['contradiction']:
                max_class = 'entailment'
            else:
                max_class = 'contradiction'

        return max_class
        # if probabilities['entailment'] > threshold:
        #     return 'entailment'
        # elif probabilities['contradiction'] > threshold:
        #     return 'contradiction'
        # else:
        #     if probabilities['neutral'] > threshold:
        #         return 'neutral'
        #     else:
        #         probabilities = {
        #             'entailment': probabilities['entailment'],
        #             'neutral': probabilities['neutral'],
        #             'contradiction': probabilities['contradiction']
        #         }
        #         max_class = max(probabilities, key=probabilities.get)
        #         return max_class

    # Function to get the string label from the predicted class ID
    def get_label_from_id(self, class_id):
        labels = self.model.config.id2label
        return labels[class_id]        

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.
        # print("logits ", logits)

        # Get model predictions
        #outputs = model(**inputs)

        # Obtain the predicted class ID and label
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_label = self.get_label_from_id(predicted_class_id)

        # Obtain the predicted class ID and label
        # predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        # predicted_label = self.get_label_from_id(predicted_class_id)
            
        # Calculate the confidence of the prediction
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence = probabilities[0][predicted_class_id].item()

        # print("Prediction Class ID: ", predicted_class_id, " Predicted Label: ", predicted_label, " Confidence: ", confidence)

        #logits_softmax = torch.softmax(logits[0], -1).tolist()
        #print("logits softmax ", logits_softmax)
        #label_names = ["entailment", "neutral", "contradiction"]
        #prediction = {name: round(float(pred), 1) for pred, name in zip(logits_softmax, label_names)}
        # print("prediction ", prediction)

        # prediction = self.classify_entailment(prediction, threshold=0.50)

        # print(prediction)

        # probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist() # torch.softmax(outputs, dim=-1) # torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
        # print("logits probs ", probs)

        # labels = ['contradiction', 'neutral', 'entailment']
        # probabilities = dict(zip(labels, logits))
        # print("probabilities ", probabilities)

        # prediction = self.classify_entailment(probabilities)
        # print("prediction ", prediction)

        # predictions = torch.softmax(logits, dim=-1)
        # print("predictions ", predictions)
        # labels = ['contradiction', 'neutral', 'entailment']
        # prediction = labels[torch.argmax(predictions, dim=-1)]
        # print("prediction ", prediction)

        # return prediction

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()
  
        return predicted_label, confidence # , probabilities


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):

    def cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_a = np.linalg.norm(vector1)
        norm_b = np.linalg.norm(vector2)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    # def jaccard_similarity(self, vector1, vector2):
    #     intersection = np.sum(np.minimum(vector1, vector2))
    #     union = np.sum(np.maximum(vector1, vector2))
    #     similarity = intersection / union if union else 0
    #     return similarity
    
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

    def predict(self, fact: str, passages: List[dict]) -> str:
        tokenized_facts = nltk.word_tokenize(fact)

        stem: bool = False # True
        remove_punctuation : bool = True
        remove_stop_words: bool = False
        similarity_metric: str = 'overlap'
        threshold: float = 0.60

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
        
        # threshold Checking
        meets_threshold = average_similarity >= threshold
        prediction = ""
        if meets_threshold:
            prediction = "S"
        else:
            prediction = "NS"

        return prediction

# class FactChecker:

#     def __init__(self, model, tokenizer, threshold=0.6):
#         # model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
#         self.tokenizer = tokenizer # AutoTokenizer.from_pretrained(model_name)
#         self.model = model # AutoModelForSequenceClassification.from_pretrained(model_name)
#         self.model.eval()
#         self.threshold = threshold

#     def get_entailment_score(self, premise, hypothesis):
#         inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             entailment_label_id = self.model.config.label2id["entailment"]
#             entailment_prob = probs[0][entailment_label_id].item()
#             # entailment_prob = probs[0][2].item()  # Assuming the entailment label index is 2
#         return entailment_prob

#     def is_sentence_entailed(self, document, summary_sentence):
#         doc_sentences = document.split('.')
#         max_score = 0
#         for sent in doc_sentences:
#             score = self.get_entailment_score(sent, summary_sentence)
#             if score > max_score:
#                 max_score = score
#         print("max score ", max_score)
#         print("threshold ", self.threshold)
#         return max_score > self.threshold

#     def is_summary_consistent(self, document, summary, consistency_threshold=0.8):
#         summary_sentences = summary.split('.')
#         entailed_sentences = 0
#         total_sentences = 0

#         for sent in summary_sentences:
#             if sent.strip():  # Ensure the sentence is not empty
#                 total_sentences += 1
#                 if self.is_sentence_entailed(document, sent):
#                     entailed_sentences += 1

#         return (entailed_sentences / total_sentences) >= consistency_threshold
    
#     def get_max_entailment_score(self, document, summary_sentence):
#         doc_sentences = document.split('.')
#         max_score = 0
#         for sent in doc_sentences:
#             score = self.get_entailment_score(sent, summary_sentence)
#             if score > max_score:
#                 max_score = score
#         return max_score

#     def is_statement_supported_by_passage(self, passage, statement):
#         passage_sentences = passage.split('.')
#         max_score = 0
#         for sent in passage_sentences:
#             score = self.get_entailment_score(sent, statement)
#             if score > max_score:
#                 max_score = score
#         return max_score > self.threshold

#     def is_fact_supported_by_passages(self, passages, fact):
#         fact_statements = fact.split('.')
#         for stmt in fact_statements:
#             supported = False
#             for passage_dict in passages:
#                 passage_text = passage_dict['text']
#                 if self.is_statement_supported_by_passage(passage_text, stmt):
#                     supported = True
#                     break
#             if not supported:
#                 return False
#         return True

# class FactChecker:

#     def __init__(self, model, tokenizer, threshold=0.7):
#         self.tokenizer = tokenizer
#         self.model = model
#         self.model.eval()
#         self.threshold = threshold

#     def get_entailment_scores(self, premises, hypothesis):
#         inputs = self.tokenizer(premises, [hypothesis for _ in premises], return_tensors="pt", truncation=True, padding='longest', max_length=512)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             entailment_label_id = self.model.config.label2id["entailment"]
#             entailment_probs = probs[:, entailment_label_id].tolist()
#         return entailment_probs

#     def is_sentence_entailed(self, doc_sentences, summary_sentence):
#         scores = self.get_entailment_scores(doc_sentences, summary_sentence)
#         max_score = max(scores)
#         return max_score > self.threshold

#     def is_summary_consistent(self, doc_sentences, summary_sentences, consistency_threshold=0.7):
#         entailed_sentences = sum([self.is_sentence_entailed(doc_sentences, sent) for sent in summary_sentences if sent.strip()])
#         return (entailed_sentences / len(summary_sentences)) >= consistency_threshold

#     def is_fact_supported_by_passages(self, passages, fact_statements):
#         # Pre-split passages into sentences
#         all_passage_sentences = [passage_dict['text'].split('.') for passage_dict in passages]
        
#         for stmt in fact_statements:
#             supported = False
#             for passage_sentences in all_passage_sentences:
#                 if self.is_sentence_entailed(passage_sentences, stmt):
#                     supported = True
#                     break
#             if not supported:
#                 return False
#         return True

class MyFactChecker:

    def __init__(self, model, tokenizer, threshold=0.5):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.threshold = threshold

    def clean_text(self, text: str) -> str:
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.replace("</s>", "")
        text = text.replace("<s>", "")
        return text

    def get_entailment_scores(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_prob, max_label = torch.max(probs, dim=1)
            max_label_str = self.model.config.id2label[max_label.item()]
        return max_label_str, max_prob.item()

    def is_sentence_entailed(self, doc_sentence, summary_sentence):
        label, score = self.get_entailment_scores(doc_sentence, summary_sentence)
        return label == "entailment" and score > self.threshold

    def is_summary_consistent(self, doc_sentences, summary_sentences, consistency_threshold=0.5):
        entailed_sentences = sum([self.is_sentence_entailed(doc_sentence, sent) for doc_sentence in doc_sentences for sent in summary_sentences if sent.strip()])
        return (entailed_sentences / len(summary_sentences)) >= consistency_threshold

    def is_fact_supported_by_passages(self, passages, fact_statements):
        # Clean and pre-split passages into sentences
        all_passage_sentences = [self.clean_text(passage_dict['text']).split('.') for passage_dict in passages]
        
        cleaned_fact_statements = [self.clean_text(stmt) for stmt in fact_statements]
        
        for stmt in cleaned_fact_statements:
            supported = False
            for passage_sentences in all_passage_sentences:
                for passage_sentence in passage_sentences:
                    if self.is_sentence_entailed(passage_sentence, stmt):
                        supported = True
                        break
                if supported:
                    break
            if not supported:
                return False
        return True

    
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


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        # print("model that is being passed in here is: ", type(ent_model))
        self.ent_model : EntailmentModel = ent_model
        self.word_recall_fact_checker = WordRecallThresholdFactChecker()
        self.checker = MyFactChecker(self.ent_model.model, self.ent_model.tokenizer, threshold=0.70)

        # self.tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        # model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

    # def clean_text(self, text: str) -> str:
    #     text = text.lower()  # Convert to lowercase
    #     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    #     # text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    #     text = text.replace("</s>", "") # re.sub(r'[^\w\s]', '', text)
    #     text = text.replace("<s>", "")
    #     return text

    # def clean_text(self, text):
    #     text = text.lower()  # Convert to lowercase
    #     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    #     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    #     return text    

    def clean_text(self, text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    # def clean_text(self, text: str) -> str:
    #     # Start with the NLTK English stop words
    #     wiki_stopwords = set(stopwords.words('english'))

    #     # Add custom Wikipedia-specific stop words
    #     # wiki_stopwords.update([
    #     #     'citation', 'needed', 'external', 'links', 'references', 'see', 'also', 
    #     #     'isbn', 'doi', 'pmid', 'retrieved', 'archive', 'url', 'accessdate', 
    #     #     'web', 'date', 'https', 'http', 'www', 'com', 'org', 'net', 'html', 
    #     #     'pdf', 'jpg', 'png'
    #     # ])

    #     # Step 1: Lowercasing
    #     text = text.lower()
        
    #     # text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    #     text = re.sub(r'<s>|</s>', '', text)  # Remove <s> and </s> tags

    #     # Step 2: Keep only alpha a-z characters
    #     # text = re.sub('[^a-z\s]', '', text)
    #     # Step 3: Stop words removal
    #     #text = ' '.join([word for word in text.split() if word not in wiki_stopwords])
    #     # Step 4: Stemming
    #     #text = ' '.join([nltk.stem.PorterStemmer().stem(word) for word in text.split()])
    #     # Step 5: Remove punctuation characters (though they should already be removed by step 2)
    #     text = re.sub(f"[{string.punctuation}]", "", text)
    #     return text

    # def evaluate_passage(self, fact, passage_text):
    #     # Clean the passage
    #     passage_text = self.clean_text(passage_text)
            
    #     # Tokenize the fact and passage and obtain model inputs
    #     #inputs = tokenizer(fact, passage_text, return_tensors="pt", truncation=True, max_length=512)

    #     # Get model predictions
    #     #outputs = model(**inputs)

    #     # Obtain the predicted class ID and label
    #     #predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    #     #predicted_label = get_label_from_id(predicted_class_id)
            
    #     # Calculate the confidence of the prediction
    #     #probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    #     #confidence = probabilities[0][predicted_class_id].item()
            
    #     return predicted_label, confidence

    def get_final_assessment2(self, fact, passages, support_threshold) -> str:

        # first use the overlap prediction model, if it doesn't pass that then throw it out
        word_overlap_prediction = self.word_recall_fact_checker.predict(fact, passages)
        if word_overlap_prediction == "NS":
            # print("throwing away a non supported prediction from the word overlap ", word_overlap_prediction)
            return "NS"
        
        # print("inside of get_final_assessment2 ", fact)
        # Thresholds for decision-making
        SUPPORT_THRESHOLD = support_threshold # 0.70 # 0.85 # 0.75 # 0.50 # 0.60
        # CONTRADICTION_THRESHOLD = 0.7

        support_confidences = []
        contradiction_confidences = []

        cleaned_fact = self.clean_text(fact)
        for passage in passages:
            # Use the entire passage text for evaluation
            # predicted_label, confidence = evaluate_passage(fact, passage['text'])
            sentence = self.clean_text(passage['text'])
            predicted_label, confidence = self.ent_model.check_entailment(cleaned_fact, sentence)
            
            # Print the result
            # print(f"Fact: '{fact}'\nPassage: '{passage['text']}'\nPrediction: {predicted_label.capitalize()} (Confidence: {confidence:.2f})\n")
            
            # Store confidence scores based on prediction
            if predicted_label == "entailment":
                support_confidences.append(confidence)
                # print("adding to the support confidence ")
            elif predicted_label == "contradiction":
                contradiction_confidences.append(confidence)
            else:
                if word_overlap_prediction == "NS":
                    # print("throwing away a non supported prediction from the word overlap ", word_overlap_prediction)
                    contradiction_confidences.append(confidence)
                else:
                    support_confidences.append(confidence)

                # print("adding to the contradiction confidence ")
            # elif predicted_label == "neutral":
            #     if confidence
                # contradiction_confidences.append(confidence)
                # print("adding to the contradiction confidence ")

        # Calculate average confidences
        #avg_support_confidence = sum(support_confidences) / len(support_confidences) if support_confidences else 0
        #avg_contradiction_confidence = sum(contradiction_confidences) / len(contradiction_confidences) if contradiction_confidences else 0

        # Calculate the average support confidence
        avg_support_confidence = sum(support_confidences) / len(support_confidences) if support_confidences else 0
        # print("avg_support_confidence ", avg_support_confidence)

        # Final decision based on average support confidence and threshold
        # if avg_support_confidence > SUPPORT_THRESHOLD:
        #     print("Final Assessment: Supported")
        # else:
        #     print("Final Assessment: Not Supported")

        # print(f"Average Support Confidence: {avg_support_confidence:.2f}")
        #print(f"Average Contradiction Confidence: {avg_contradiction_confidence:.2f}")

        # Final decision based on average confidences and thresholds
        # print("Avg support confidence vs support threshold ", avg_support_confidence)
        if avg_support_confidence > SUPPORT_THRESHOLD: # and avg_support_confidence > avg_contradiction_confidence:
            # print("Final Assessment: Supported")
            return "S"
        else:
            # print("Final Assessment: Not Supported")
            return "NS"

    def predict(self, fact: str, passages: List[dict]) -> str:

        # first use the overlap prediction model, if it doesn't pass that then throw it out
        word_overlap_similarity = self.word_recall_fact_checker.evaluate_similarity(fact, passages)
        if word_overlap_similarity < 0.50:
            # print("throwing away a non supported prediction from the word overlap ", word_overlap_prediction)
            return "NS"
        
        threshold = 0.70

        final_assessment = self.get_final_assessment2(fact, passages, support_threshold=threshold)
        return final_assessment
    
        # result = self.checker.is_fact_supported_by_passages(passages, fact)
        # print(f"The fact is {'supported' if result else 'not supported'} by the passages.")

        if final_assessment == result and result == "NS":
            return "NS"
        else:
            return "S"

        cleaned_fact = self.clean_text(fact)
        predictions = []
        
        support_count = 0
        total_passages = len(passages)
        # print("INSIDE predict!!!! ", fact)
        passage_predictions = []

        # print("passages ", passages)
        for passage in passages:
            # clean and split the text into sentences

            cleaned_text = self.clean_text(passage["text"])
            # cleaned_text = self.clean_text(passage["title"] + "." + passage["text"])

            # print("TITLE: ", passage["title"])
            # print("TEXT: ", passage["text"])

            sentences = nltk.sent_tokenize(cleaned_text)

            #max_prediction = "contradiction"  # Initialize with the minimum value
            
            # Dictionary to keep track of prediction counts for each label
            prediction_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}

            #print("# of sentences ", len(sentences))
            #for sent in sentences:
            #    print("sent: ", sent, "\n")
            #print("sent token BEFORE ", cleaned_text, " sent token AFTER ", sentences)

            # Loop over the sentences
            for sentence in sentences:
                # use the model here
                prediction = self.ent_model.check_entailment(cleaned_fact, sentence)
                
                # Update the prediction counts
                prediction_counts[prediction] += 1

                # # Update the max prediction if the current sentence entails the fact
                # if prediction == "entailment":
                #     max_prediction = "entailment"
                #     break  # No need to check further if we already found an entailment

                # if prediction == "entailment":
                #     support_count += 1

                # print("cleaned_fact: ", cleaned_fact, " sentence: ", sentence, " prediction: ", prediction)
                #predictions.append(prediction)
            
            # Determine the max prediction based on majority count
            # max_prediction = max(prediction_counts, key=lambda k: (prediction_counts[k], k == "entailment"))

            # Determine the max prediction for the passage based on majority count
            max_prediction = max(prediction_counts, key=lambda k: (prediction_counts[k], k == "entailment"))
            passage_predictions.append(max_prediction)

            # Print the result
            #print(f"Fact: '{fact}'\nPassage: '{passage}'\nMax Prediction: {max_prediction.capitalize()}\n")
    
            # Count the number of supports (entailments)
            #if max_prediction == "entailment":
            #    support_count += 1

        """
        # Initialize variables to store the max confidence and associated prediction type
        max_confidence = -1
        max_pred = None

        # Iterate over each dictionary in the predictions list
        for prediction_confidences in predictions:
            # Iterate over each prediction type and its confidence score in the current dictionary
            for prediction, confidence in prediction_confidences.items():
                # If the confidence is greater than the current max confidence, update max confidence and max pred
                if confidence > max_confidence:
                    max_confidence = confidence
                    max_pred = prediction
                elif confidence == max_confidence and max_pred == 'neutral' and prediction != 'neutral':
                    max_pred = prediction

        # Special case to handle ties when 'neutral' has the highest confidence
        if max_pred == 'neutral':
            # If 'entailment' and 'contradiction' have equal confidence, assign 'contradiction'
            if predictions[-1]['entailment'] == predictions[-1]['contradiction']:
                max_pred = 'contradiction'
            # Otherwise, assign the prediction with higher confidence between 'entailment' and 'contradiction'
            elif predictions[-1]['entailment'] > predictions[-1]['contradiction']:
                max_pred = 'entailment'
            else:
                max_pred = 'contradiction'
        """
        
        # print("Fact: ", fact)
        # print("Passages: ", passages)

        # Determine the final assessment based on the majority prediction of all passages
        final_prediction_counts = {"entailment": passage_predictions.count("entailment"), 
                                "neutral": passage_predictions.count("neutral"), 
                                "contradiction": passage_predictions.count("contradiction")}
        final_assessment = max(final_prediction_counts, key=lambda k: (final_prediction_counts[k], k == "entailment"))

        # Print the final binary assessment
        if final_assessment == "entailment":
            #print("Final Assessment: Supported")
            final_prediction="S"
        elif final_assessment == "contradiction":
            #print("Final Assessment: Not Supported")
            final_prediction="NS"
        elif final_assessment == "neutral":
            if final_prediction_counts["entailment"] >= final_prediction_counts["contradiction"]:
                final_prediction="S"
            else:
                final_prediction="NS"
        else:
            final_prediction="S"

        # if (support_count > 0): #or ((support_count == (total_passages / 2)) and ((total_passages % 2) == 0)):
        #     # print("Final Assessment: Supported")
        #     final_prediction="S"
        # else:
        #     # print("Final Assessment: Not Supported")
        #     final_prediction="NS"

        # if (support_count > (total_passages / 2)) or ((support_count == (total_passages / 2)) and ((total_passages % 2) == 0)):
        #     # print("Final Assessment: Supported")
        #     final_prediction="S"
        # else:
        #     # print("Final Assessment: Not Supported")
        #     final_prediction="NS"

        # final_prediction = ""
        # if max_pred == 'entailment' and max_confidence > 0.75:
        #     final_prediction="S"
        # else:
        #     final_prediction="NS"
        return final_prediction

# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

