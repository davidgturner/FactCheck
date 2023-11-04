# factcheck.py

import re
import string
import torch
import numpy as np
import spacy
import gc

import nltk
from typing import List
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk import pos_tag

from transformers import AutoConfig

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

spacy_nlp = spacy.load("en_core_web_sm")

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
        config = AutoConfig.from_pretrained(self.model.config.name_or_path)
        self.label_mapping = config.id2label

    # get_label_from_id - function to get the string label from the predicted class ID
    def get_label_from_id(self, class_id):
        labels = self.label_mapping
        return labels[class_id]        

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            max_token_length = 512
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True, max_length=max_token_length)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()
  
        return probs

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

    def word_overlap(self, tokenized_facts: str, tokenized_passage: str):
        numerator = len(set(tokenized_facts) & set(tokenized_passage))
        denominator = min(len(set(tokenized_facts)), len(set(tokenized_passage)))
        overlap_coefficient = numerator / denominator
        return overlap_coefficient
    
    def evaluate_similarity(self, fact, passages) -> float:
        tokenized_facts = nltk.word_tokenize(fact)

        stem: bool = False
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

        # remove empty strings
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
                sim = self.word_overlap(tokenized_facts, tokenized_passage)
            else:
                raise ValueError(f'Unsupported similarity metric: {similarity_metric}')
        
            # add similarity result to results list
            results.append(sim)

        # compute average similarity score
        average_similarity = np.average(results) if results else 0
        return average_similarity

    def predict(self, fact: str, passages: List[dict]) -> str:
        threshold: float = 0.60

        # compute average similarity score
        word_overlap_similarity = self.evaluate_similarity(fact, passages)
        
        # threshold Checking
        meets_threshold = word_overlap_similarity >= threshold
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
        self.sent_tokenizer = PunktSentenceTokenizer()

    def clean_text(self, text):
        text = re.sub(r'<s>|</s>', '', text)  # remove <s> and </s> tokens
        return text

    def chunk_text(self, text, max_length):
        # clean text first
        text = self.clean_text(text)
        
        # use tokenizer to encode the text, handling padding and truncation
        encoded_input = self.ent_model.tokenizer(
            text,
            add_special_tokens=False,
            max_length=max_length,   # specfiy max length for each chunk
            padding='max_length',    # pad max_length
            truncation=True,         # truncate max_length
            return_tensors='pt'      
        )
        
        # convert token IDs back to tokens to get a list of strings
        tokens = self.ent_model.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0], skip_special_tokens=True)
        
        # convert tokens to a single string
        text_chunk = self.ent_model.tokenizer.convert_tokens_to_string(tokens)
        clean_passage_text = self.clean_text(text_chunk)

        # Use the PunktSentenceTokenizer to split the text into sentences
        sentences = self.sent_tokenizer.tokenize(clean_passage_text)

        return sentences

# from concurrent.futures import ThreadPoolExecutor, as_completed

    def check_fact(self, fact, passages, threshold=0.20, max_length=512):
        max_entailment_score = 0.0
        most_entailing_sentence = ""
        max_contradiction_score = 0.0
        most_contradicting_sentence = ""
        evaluated_passages = []
        word_overlap_decision = self.word_recall_fact_checker.predict(fact, passages)

        for passage in passages:
            full_text = passage['title'] + " is about " + passage['text']
            clean_text = self.clean_text(full_text)

            sentences = self.chunk_text(clean_text, max_length)
            passage_evaluations = {}

            for sentence in sentences:
                entailment_prob, neutral_prob, contradiction_prob = self.ent_model.check_entailment(fact, sentence)
                evaluation = "S" if entailment_prob > threshold else "NS"
                passage_evaluations[sentence] = evaluation

                if entailment_prob > max_entailment_score:
                    max_entailment_score = entailment_prob
                    most_entailing_sentence = sentence
                    
                if contradiction_prob > max_contradiction_score:
                    max_contradiction_score = contradiction_prob
                    most_contradicting_sentence = sentence

            passage_eval_result = "S" if any(evaluation == "S" for evaluation in passage_evaluations.values()) else "NS"
            evaluated_passages.append({"passage": passage_evaluations, "passage_eval_result": passage_eval_result})

        decision = "S" if any(passage["passage_eval_result"] == "S" for passage in evaluated_passages) else word_overlap_decision

        return {
            "decision": decision,
            "max_entailment_score": max_entailment_score,
            "most_entailing_sentence": most_entailing_sentence,
            "max_contradiction_score": max_contradiction_score,
            "most_contradicting_sentence": most_contradicting_sentence,
            "evaluated_passages": evaluated_passages
        }

    
    def predict(self, fact: str, passages: List[dict], overlap_threshold=0.40, positive_threshold=0.20) -> str:
        max_length = 512
        word_overlap_similarity = self.word_recall_fact_checker.evaluate_similarity(fact, passages)
        if word_overlap_similarity < overlap_threshold:
            return "NS"
        
        cleaned_fact = self.clean_text(fact)
        result_c = self.check_fact(cleaned_fact, passages, threshold=positive_threshold, max_length=max_length)
        decision_c = result_c["decision"]

        return decision_c

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

