# factcheck.py

import re
import string
import torch
import numpy as np
import spacy
import gc

import nltk
from typing import List, Dict


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
        self.label_mapping = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

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

    # get_label_from_id - function to get the string label from the predicted class ID
    def get_label_from_id(self, class_id):
        # print("get_label_from_id input class_id ", class_id)
        # print("get_label_from_id label dict ", self.model.config.id2label)
        # labels = self.model.config.id2label
        labels = self.label_mapping
        return labels[class_id]        

    def check_entailment(self, premise: str, hypothesis: str):

        # Ensure model is in evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # print("logits ", logits)

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # obtain the predicted class ID and label
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_label = self.get_label_from_id(predicted_class_id)

        # calculate the confidence of the prediction
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        # print("probs ", probabilities)
        confidence = probabilities[0][predicted_class_id].item()



        # Get predicted class and confidence
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_label = self.get_label_from_id(predicted_class_id)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence = probabilities[0][predicted_class_id].item()

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()
  
        return predicted_label, confidence


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
        
            # similarity calculation
            if similarity_metric == 'overlap':
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

        stem: bool = False
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
        
            # similarity calculation
            if similarity_metric == 'overlap':
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
class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model : EntailmentModel = ent_model
        self.word_recall_fact_checker = WordRecallThresholdFactChecker()
        self.threshold = .70

    def clean_text(self, text: str) -> str:
        # convert to lowercase
        text = text.lower()
        
        # replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # remove <s> and </s> tags
        text = re.sub(r'<s>|</s>', '', text)
        
        # remove punctuation
        text = re.sub(f"[{string.punctuation}]", "", text)
        
        # Uncomment below if you want to remove stopwords and apply stemming
        # wiki_stopwords = set(stopwords.words('english'))
        # text = ' '.join([word for word in text.split() if word not in wiki_stopwords])
        # text = ' '.join([nltk.stem.PorterStemmer().stem(word) for word in text.split()])
        
        return text

    def get_final_assessment(self, fact, passages, support_threshold=0.70) -> str:

        support_confidences = []
        contradiction_confidences = []

        for passage in passages:
            # Combine title and text, then clean and tokenize
            full_text = passage['title'] + ". " + passage['text']
            sentences = nltk.sent_tokenize(full_text)
            
            # Filter out sentences that are 1-2 words or only numeric/punctuation characters
            filtered_sentences = [sentence for sentence in sentences if len(sentence.split()) > 2 and not sentence.isnumeric() and not all(char in string.punctuation for char in sentence)]

            # Use list comprehension to get predictions for all sentences
            predictions = [self.ent_model.check_entailment(fact, self.clean_text(sentence)) for sentence in filtered_sentences]

            # Update confidence lists based on predictions
            support_confidences.extend([conf for label, conf in predictions if label == "entailment"])
            contradiction_confidences.extend([conf for label, conf in predictions if label == "contradiction"])

            if len(contradiction_confidences) > 0:
                return "NS"
            # neutral_confidences.extend([conf for label, conf in predictions if label == "neutral"])

        # Calculate average confidences
        avg_support_confidence = sum(support_confidences) / len(support_confidences) if support_confidences else 0
        avg_contradiction_confidence = sum(contradiction_confidences) / len(contradiction_confidences) if contradiction_confidences else 0

        # Make final decision based on average confidences and thresholds
        if avg_support_confidence > support_threshold and avg_support_confidence > avg_contradiction_confidence and avg_contradiction_confidence == 0.0:
            return "S"
        else:
            return "NS"


    # def get_final_assessment(self, fact, passages, support_threshold=0.70) -> str:

    #     # # first use the overlap prediction model, if it doesn't pass that then throw it out
    #     # word_overlap_prediction = self.word_recall_fact_checker.predict(fact, passages)
    #     # if word_overlap_prediction == "NS":
    #     #     return "NS"
        
    #     # threshold for decision-making
    #     SUPPORT_THRESHOLD = support_threshold

    #     support_confidences = []
    #     contradiction_confidences = []
    #     neutral_confidences = []

    #     # cleaned_fact = self.clean_text(fact)
    #     #print("passages ", passages)
    #     for passage in passages:
    #         # sentence tokenize
    #         sentences = nltk.sent_tokenize(passage['title'] + ". " + passage['text'])

    #         #print("sentences: ", sentences)
    #         #self.clean_text(
    #         for sentence in sentences:
    #             predicted_label, confidence = self.ent_model.check_entailment(fact, self.clean_text(sentence))
                
    #             # store confidence scores based on prediction
    #             if predicted_label == "entailment":
    #                 support_confidences.append(confidence)
    #             elif predicted_label == "contradiction":
    #                 contradiction_confidences.append(confidence)
    #             elif predicted_label == "neutral":
    #                 neutral_confidences.append(confidence)

    #     # calculate the average support confidence
    #     avg_support_confidence = sum(support_confidences) / len(support_confidences) if support_confidences else 0
    #     avg_contradiction_confidence = sum(contradiction_confidences) / len(contradiction_confidences) if contradiction_confidences else 0
    #     #avg_neutral_confidence = sum(neutral_confidences) / len(neutral_confidences) if neutral_confidences else 0

    #     # cinal decision based on average confidences and thresholds
    #     if avg_support_confidence > 0.0 and avg_support_confidence > avg_contradiction_confidence and avg_contradiction_confidence == 0.0:
    #         return "S"
    #     else:
    #         return "NS"

    def predict(self, fact: str, passages: List[dict]) -> str:
        word_overlap_similarity = self.word_recall_fact_checker.evaluate_similarity(fact, passages)
        if word_overlap_similarity < 0.20:
            return "NS"
        cleaned_fact = self.clean_text(fact)
        
        # call the consolidated function to get the final assessment
        final_assessment = self.get_final_assessment(cleaned_fact, passages, 0.70)
    
        return final_assessment

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

