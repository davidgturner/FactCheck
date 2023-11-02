# factscore_retrieval_interface.py

import json
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from tqdm import tqdm
import argparse
from factcheck import *

import csv
from tqdm import tqdm

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help="choose from [random', 'always_entail', 'word_overlap', 'parsing', 'entailment]")
    # parser.add_argument('--labels_path', type=str, default="data/labeled_ChatGPT.jsonl", help="path to the labels")
    parser.add_argument('--labels_path', type=str, default="data/dev_labeled_ChatGPT.jsonl", help="path to the labels")
    parser.add_argument('--passages_path', type=str, default="data/passages_bm25_ChatGPT_humfacts.jsonl", help="path to the passages retrieved for the ChatGPT human-labeled facts")
    args = parser.parse_args()
    return args


def read_passages(path: str):
    """
    Reads the retrieved passages and puts then in a dictionary mapping facts to passages.
    :param path: path to the cached passages
    :return: dict mapping facts (strings) to passages
    """
    fact_to_passage_dict = {}
    with open(path, 'r') as file:
        all_lines = file.readlines()
        for nextline in all_lines:
            dict = json.loads(nextline)
            name = dict["name"]
            for passage in dict["passages"]:
                if passage["title"] != name:
                    raise Exception("Couldn't find a match for: " + name + " " + passage["title"])
            fact_to_passage_dict[dict["sent"]] = dict["passages"]
    return fact_to_passage_dict


def read_fact_examples(labeled_facts_path: str, fact_to_passage_dict: Dict):
    """
    Reads the labeled fact examples and constructs FactExample objects associating labeled, human-annotated facts
    with their corresponding passages
    :param labeled_facts_path: path to the list of labeled
    :param fact_to_passage_dict: the dict mapping facts to passages (see load_passages)
    :return: a list of FactExample objects to use as our dataset
    """
    examples = []
    with open(labeled_facts_path, 'r') as file:
        all_lines = file.readlines()
        for nextline in all_lines:
            dict = json.loads(nextline)
            if dict["annotations"] is not None:
                for sent in dict["annotations"]:
                    if sent["human-atomic-facts"] is None:
                        # Should never be the case, but just in case
                        print("No facts! Skipping this one: " + repr(sent))
                    else:
                        for fact in sent["human-atomic-facts"]:
                            if fact["text"] not in fact_to_passage_dict:
                                # Should never be the case, but just in case
                                print("Missing fact: " + fact["text"])
                            else:
                                examples.append(FactExample(fact["text"], fact_to_passage_dict[fact["text"]], fact["label"]))
    return examples


def predict_two_classes(examples: List[FactExample], fact_checker, nt, pt):
    """
    Compares against ground truth which is just the labels S and NS (IR is mapped to NS).
    Makes predictions and prints evaluation statistics on this setting.
    :param examples: a list of FactExample objects
    :param fact_checker: the FactChecker object to use for prediction
    """
    gold_label_indexer = ["S", "NS"]
    confusion_mat = [[0, 0], [0, 0]]
    ex_count = 0

    # Setup CSV file
    with open('results.csv', 'w', newline='') as csvfile:
        #fieldnames = ['Fact', 'Clean Fact', '# of Total Passages', '# of Passage with S', '# of Passage with NS', 'Golden Label', 'Prediction Label', 'Correct?']
        fieldnames = ['Fact', '# of Total Passages', '# of Passage with S', '# of Passage with NS', 'Golden Label', 'Prediction Label', 'Correct?']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, example in enumerate(tqdm(examples)):
            converted_label = "NS" if example.label == 'IR' else example.label
            gold_label = gold_label_indexer.index(converted_label)

            raw_pred = fact_checker.predict(example.fact, example.passages, nt, pt)
            # raw_pred = fact_checker.predict(example.fact, example.passages)
            pred_label = gold_label_indexer.index(raw_pred)

            #if pred_label != gold_label:
            #    print("incorrect: example.fact: ", example.fact, " gold label ", converted_label, " pred label ", raw_pred)

            # Compute the desired metrics
            total_passages = len(example.passages)
            passages_with_s = sum(1 for p in example.passages if 'S' in p)
            passages_with_ns = total_passages - passages_with_s
            correct = 'Y' if pred_label == gold_label else 'N'

            # Write to CSV
            writer.writerow({
                'Fact': example.fact,
                # 'Clean Fact': fact_checker.clean_text(example.fact),
                '# of Total Passages': total_passages,
                '# of Passage with S': passages_with_s,
                '# of Passage with NS': passages_with_ns,
                'Golden Label': converted_label,
                'Prediction Label': raw_pred,
                'Correct?': correct
            })

            confusion_mat[gold_label][pred_label] += 1
            ex_count += 1
    return print_eval_stats(confusion_mat, gold_label_indexer)


def print_eval_stats(confusion_mat, gold_label_indexer) -> float:
    """
    Takes a confusion matrix and the label indexer and prints accuracy and per-class F1
    :param confusion_mat: The confusion matrix, indexed as [gold_label, pred_label]
    :param gold_label_indexer: The Indexer for the labels as a List, not an Indexer
    """
    for row in confusion_mat:
        print("\t".join([repr(item) for item in row]))
    correct_preds = sum([confusion_mat[i][i] for i in range(0, len(gold_label_indexer))])
    total_preds = sum([confusion_mat[i][j] for i in range(0, len(gold_label_indexer)) for j in range(0, len(gold_label_indexer))])
    print("Accuracy: " + repr(correct_preds) + "/" + repr(total_preds) + " = " + repr(correct_preds/total_preds))
    for idx in range(0, len(gold_label_indexer)):
        num_correct = confusion_mat[idx][idx]
        num_gold = sum([confusion_mat[idx][i] for i in range(0, len(gold_label_indexer))])
        num_pred = sum([confusion_mat[i][idx] for i in range(0, len(gold_label_indexer))])
        rec = num_correct / num_gold

        if num_pred > 0:
            prec = num_correct / num_pred
            if (prec + rec) > 0:
                f1 = 2 * prec * rec/(prec + rec)
            else:
                f1 = "undefined"
                prec = "undefined"
        else:
            prec = "undefined"
            f1 = "undefined"
        print("Prec for " + gold_label_indexer[idx] + ": " + repr(num_correct) + "/" + repr(num_pred) + " = " + repr(prec))
        print("Rec for " + gold_label_indexer[idx] + ": " + repr(num_correct) + "/" + repr(num_gold) + " = " + repr(rec))
        print("F1 for " + gold_label_indexer[idx] + ": " + repr(f1))

        return correct_preds/total_preds


if __name__=="__main__":
    args = _parse_args()
    print(args)

    fact_to_passage_dict = read_passages(args.passages_path)

    examples = read_fact_examples(args.labels_path, fact_to_passage_dict)
    print("Read " + repr(len(examples)) + " examples")
    print("Fact and length of passages for each fact:")
    for example in examples:
        print(example.fact + ": " + repr([len(p["text"]) for p in example.passages]))

    assert args.mode in ['random', 'always_entail', 'word_overlap', 'parsing', 'entailment'], "invalid method"
    print(f"Method: {args.mode}")

    fact_checker = None
    if args.mode == "random":
        fact_checker = RandomGuessFactChecker()
    elif args.mode == "always_entail":
        fact_checker = AlwaysEntailedFactChecker()
    elif args.mode == "word_overlap":
        fact_checker = WordRecallThresholdFactChecker()
    elif args.mode == "parsing":
        fact_checker = DependencyRecallThresholdFactChecker()
    elif args.mode == "entailment":
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        # model_name = "roberta-large-mnli"   # alternative model that you can try out if you want
        ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
        roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        ent_model = EntailmentModel(roberta_ent_model, ent_tokenizer)
        fact_checker = EntailmentFactChecker(ent_model)
    else:
        raise NotImplementedError

    nt = 0.45 # 0.35 # 0.2
    pt = 0.2 # 0.30
    # overlap = 50 # 10
    # examples = random.sample(examples, 25)
    predict_two_classes(examples, fact_checker, nt, pt)
    # predict_two_classes(examples, fact_checker)

    # Hyperparam tuning
    """
    # Define the range of thresholds
    pos_thresholds = [i * 0.05 for i in range(2, 13)]  # 0.10, 0.15, ..., 0.60
    neg_thresholds = [i * 0.05 for i in range(2, 11)]  # 0.10, 0.15, ..., 0.50

    best_accuracy = 0.0
    best_pos_threshold = 0.0
    best_neg_threshold = 0.0

    # Loop over the thresholds
    for pt in pos_thresholds:
        for nt in neg_thresholds:
            # Sample a subset of examples for testing
            sampled_examples = random.sample(examples, 25)
            
            # Calculate accuracy for the current thresholds
            accuracy = predict_two_classes(sampled_examples, fact_checker, nt, pt)
            
            # Update best thresholds if current accuracy is higher
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pos_threshold = pt
                best_neg_threshold = nt

                print("best so far:======")
                print("Best Positive Threshold:", best_pos_threshold)
                print("Best Negative Threshold:", best_neg_threshold)
                print("Best Accuracy:", best_accuracy)
    """

    # # Print the best thresholds
    # print("Best Positive Threshold:", best_pos_threshold)
    # print("Best Negative Threshold:", best_neg_threshold)
    # print("Best Accuracy:", best_accuracy)

    # Define the ranges
    # pos_thresholds = [i/10 for i in range(1, 4)]  # 0.1, 0.2, 0.3
    # neg_thresholds = [i/10 for i in range(2, 5)]  # 0.2, 0.3, 0.4
    # overlap_values = list(range(5, 55, 5))  # 5, 10, ..., 50

    # # Define the ranges
    # pos_thresholds = [i/10 for i in range(1, 8)]  # 0.1, 0.2, ..., 0.7
    # neg_thresholds = [i/10 for i in range(1, 6)]  # 0.1, 0.2, ..., 0.5
    # overlap_values = list(range(5, 55, 5))  # 5, 10, ..., 50

    # best_accuracy = 0.0
    # best_pos_threshold = 0.0
    # best_neg_threshold = 0.0
    # best_overlap = 0

    # for pt in pos_thresholds:
    #     for nt in neg_thresholds:
    #         for ov in overlap_values:
    #             # Sample a subset of examples for testing
    #             sampled_examples = random.sample(examples, 30)

    #             accuracy = predict_two_classes(sampled_examples, fact_checker, nt, pt, ov)
    #             if accuracy > best_accuracy:
    #                 best_accuracy = accuracy
    #                 best_pos_threshold = pt
    #                 best_neg_threshold = nt
    #                 best_overlap = ov

    #                 print("best so far:======")
    #                 print("Best Positive Threshold:", best_pos_threshold)
    #                 print("Best Negative Threshold:", best_neg_threshold)
    #                 print("Best Overlap:", best_overlap)
    #                 print("Best Accuracy:", best_accuracy)

    # print("Best Positive Threshold:", best_pos_threshold)
    # print("Best Negative Threshold:", best_neg_threshold)
    # print("Best Overlap:", best_overlap)
    # print("Best Accuracy:", best_accuracy)
