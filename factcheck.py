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
        print("ENTAILMENT MODEL TYPE THIS DERIVES FROM ", type(self.model))

    def classify_entailment(self, probabilities, threshold=0.5):
        if probabilities['entailment'] > threshold:
            return 'entailment'
        elif probabilities['contradiction'] > threshold:
            return 'contradiction'
        else:
            return 'neutral'

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()

        labels = ['contradiction', 'neutral', 'entailment']
        probabilities = dict(zip(labels, probs))
        prediction = self.classify_entailment(probabilities)

        # predictions = torch.softmax(logits, dim=-1)
        # labels = ['contradiction', 'neutral', 'entailment']
        # prediction = labels[torch.argmax(predictions, dim=-1)]
        # return prediction

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        return prediction


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

class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        print("model that is being passed in here is: ", type(ent_model))
        self.ent_model : EntailmentModel = ent_model

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text

    def jaccard_similarity(self, str1: str, str2: str) -> float:
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def predict(self, fact: str, passages: List[dict]) -> str:
        cleaned_fact = self.clean_text(fact)

        max_similarity = 0
        most_similar_sentence = ""

        predictions = []

        for passage in passages:
            # clean and split the text into sentences
            cleaned_text = self.clean_text(passage["text"])
            sentences = nltk.sent_tokenize(cleaned_text)

            # Loop over the sentences
            for sentence in sentences:
                # use the model here
                prediction = self.ent_model.check_entailment(cleaned_fact, sentence)
                predictions.append(prediction)
                # Compute the similarity between the sentence and the given fact
                # similarity = self.jaccard_similarity(sentence, cleaned_fact)

                # update max similarity and the most similar sentence
                # if similarity > max_similarity:
                #     print("sim sim ", similarity, " for sent ", sentence)
                #     max_similarity = similarity
                #     most_similar_sentence = sentence

            # count occurrences of each prediction type
            prediction_counts = Counter(predictions)

            # sort predictions by the count and the priority order
            sorted_predictions = sorted(prediction_counts.items(), key=lambda x: (-x[1], ['entailment', 'neutral', 'contradiction'].index(x[0])))

            # return prediction with the maximum count and highest priority
            max_pred = sorted_predictions[0][0]
            # print("max pred ", max_pred)
            preddy = ""
            if max_pred == 'entailment':
                preddy="S"
            else:
                preddy="NS"
            return preddy

        # return most_similar_sentence, max_similarity

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

