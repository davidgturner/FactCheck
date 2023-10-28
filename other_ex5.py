import json
import re
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from factcheck import EntailmentFactChecker, EntailmentModel

# Function to clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# # Single fact
# fact = "Children are playing soccer in the park."

# # Group of passages
# passages = [
#     "Kids are playing sports outdoors. They seem to be having a lot of fun.",
#     "The children are at school. They are attending their classes.",
#     "The children are reading books indoors. It's a quiet and peaceful environment.",
#     "A group of kids is playing football in the garden. They are very energetic.",
#     "Children are running around in the park, some are playing with a ball. It's a sunny day."
# ]



passages_string = """

[{'title': 'Hibo Wardere', 'text': '<s>Hibo Wardere is a Somali-born campaigner against female genital mutilation (FGM), author, and public speaker. Born in Somalia, she moved to London, England when just a teenager in 1989, as a refugee fleeing the Somali Civil War. She currently resides in Walthamstow, London, where she worked as a mediator and a regular FGM educator for Waltham Forest Borough. Her testimonials and campaigning work have made her one of Britain\'s most prominent campaigners about FGM and she has appeared in numerous publications, including the "Telegraph", the BBC, and "The Guardian".</s><s>Early life. Hibo Wardere was born in Somalia. At the age of six, she was the victim of type 3 FGM, an event she has described as "being engulfed in pain from head to toe". Every day for the next ten years, she sought answers from her mother, but was always denied a response. When Wardere was 16, she finally struck a deal with a relative, who promised to tell her everything about what happened after her wedding night. She was horrified by the revelations, and soon fled to London after the civil war broke out in the 1980s.</s><s>Activism. When'}, {'title': 'Hibo Wardere', 'text': ' quoted as saying “It is a sexual abuse. It brings shame and rips women and girls of their dignity. It should be stopped". Wardere\'s main ambition for the future is to see Female Genital Mutilation eradicated in her lifetime. Her memoir, "Cut", was published in April 2016.</s><s>Personal life. lives with her husband Yusuf and their seven children.</s>'}, {'title': 'Hibo Wardere', 'text': ' she arrived in London, Wardere sought treatment for her wounds, but received little support from the NHS. Doctors failed to ask what had happened to her, and only rarely mentioned FGM on her medical files, even when she gave birth to her children. Wardere eventually found the answers she was looking for at the library, where she read about female mutilation in a book. Years later, when she was studying to become a teaching assistant, she opened up about her story in a homework essay. The head of staff read her work and asked her to deliver a speech to 120 teachers, during which some realised that their students might have experienced the same trauma. After reading Wardere\'s essay, school governor Clare Coghill booked Wardere appointments with other schools in the area. Wardere has worked as a mediator and FGM educator since then, helping young students escape FGM. She also delivers awareness raising sessions to doctors and the Police to assist in their understanding of FGM. Her testimonies have appeared in numerous publications, including the BBC, the "Guardian" and the "Telegraph". Wardere has advised that as an FGM survivor, she is aware many other women who have undergone the practice feel too ashamed to speak out about their suffering. Wardere i in London, Wardere sought treatment for her wounds, but received little support from the NHS. Doctors failed to ask what had happened to her, and only rarely mentioned FGM on her medical files, even when she gave birth to her children. Wardere eventually found the answers she was looking for at the library, where she read about female mutilation in a book. Years later, when she was studying to become a teaching assistant, she opened up about her story in a homework essay. The head of staff read her work and asked her to deliver a speech to 120 teachers, during which some realised that their students might have experienced the same trauma. After reading Wardere\'s essay, school governor Clare Coghill booked Wardere appointments with other schools in the area. Wardere has worked as a mediator and FGM educator since then, helping young students escape FGM. She also delivers awareness raising sessions to doctors and the Police to assist in their understanding of FGM. Her testimonies have appeared in numerous publications, including the BBC, the "Guardian" and the "Telegraph". Wardere has advised that as an FGM survivor, she is aware many other women who have undergone the practice feel too ashamed to speak out about their suffering. Wardere is'}]

"""

# Extract titles and texts using regex
titles = re.findall(r"'title':\s*'([^']+)'", passages_string)
texts = re.findall(r"'text':\s*'([^']+)'", passages_string)

# Build a list of dictionaries
passages = [{"title": title, "text": text} for title, text in zip(titles, texts)]

# Convert to JSON
passages_json = json.dumps(passages, indent=4)
print(passages_json)

# passages_string = """
# [{'title': 'Jean Daullé', 'text': '<s>Jean Daullé (18 May 1703 – 23 April 1763) was a French engraver.</s><s>Biography. He was the son of Jean Daullé, a silversmith, and his wife, Anne née Dennel. At the age of fourteen, he received training from an engraver named Robart, at the priory of Saint-Pierre d\'Abbeville. He then went to Paris, and worked at the studios of Robert Hecquet (1693-1775), who was also originally from Picardy. In 1735, his work attracted the attention of the engraver and merchant, Pierre-Jean Mariette, who provided him with professional recommendations. Soon after, he was approached by the painter, Hyacinthe Rigaud, who wanted to 
# make him his official engraver. In 1742, Daullé was received at the Académie Royale de Peinture et de Sculpture, with his presentation, "Hyacinthe Rigaud Painting his Wife", after a work by Rigaud. He was also admitted as a member of the academy in Augsbourg. Eventually named "Engraver to the King",'}, {'title': 'Jean Daullé', 'text': ' he trained the future publisher and print dealer, Pierre-François Basan, as well as the German engraver Jean-Georges Wille. Around 1745, he married Gabrielle-Anne Landry and they had five children. Overwhelmed by his large family, his work suffered.</s><s>Work. He engraved several portraits and plates of historical and other subjects, which are chiefly executed with the graver in a clear and firm style, which entitles him to rank with the ablest artists of his time. He marked his works J. D. The following are his principal plates:</s><s>Work.:Portraits. - "Catherine, Countess of Feuquières, daughter of Pierre Mignard"; after Mignard. - "Hyacinthe Rigaud, painter"; after Rigaud; engraved for his reception at the Academy in 1742. - "Marguerite de Valois, Countess of Caylus"; after the same. - "Charles Edward Stuart, son of the Pretender". - "Clementina, Princess of Poland, his consort"; after David. - "Madame Favart, in the part of \'Bastienne;\' " after Carle van Loo the future publisher and print dealer, Pierre-François Basan, as well as the German engraver Jean-Georges Wille. Around 1745, he married Gabrielle-Anne Landry and they had five children. Overwhelmed by his large family, his work suffered.</s><s>Work. He engraved several portraits and plates of historical and other subjects, which are chiefly executed with the graver in a clear and firm style, which entitles him to rank with the ablest artists of his time. He marked his works J. D. The following are his principal plates:</s><s>Work.:Portraits. - "Catherine, Countess of Feuquières, daughter of Pierre Mignard"; after Mignard. - "Hyacinthe Rigaud, painter"; after Rigaud; engraved for his reception at the Academy in 1742. - "Marguerite de Valois, 
# Countess of Caylus"; after the same. - "Charles Edward Stuart, son of the Pretender". - "Clementina, Princess of Poland, his consort"; after David. 
# - "Madame Favart, in the part of \'Bastienne;\' " after Carle van Loo.'}, {'title': 'Jean Daullé', 'text': " after Van Dyck. A detailed account of this artist's works is contained in Delignière's 'Catalogue raisonné de l'oeuvre gravé de d' Abbeville,' 1872, 8vo.</s><s>References. - - Attribution:</s>"}, {'title': 'Jean Daullé', 'text': ' - "Claude Deshayes Gendron, oculist"; after Rigaud. - "Jean Baptiste Rousseau"; after Aved. - "Jean Mariette, engraver"; after Pesne.</s><s>Work.:Subjects after various masters. - "The Magdalen"; after Correggio; for the Dresden Gallery. - "Diogenes with his Lantern"; after Spagnoletto; for the same. - "Quos Ego"; after Rubens. - "The Two Sons of Rubens"; after the same; for the Dresden Gallery. - "Neptune appeasing the Tempest"; after the same. - "Charity with Three Children"; after Albani. - "The Triumph of Venus"; after Boucher. - "Les Amusemens de la Campagne"; after Boucher. - "Latona"; after J. Jouvenet. - "Four Marine subjects"; after Joseph Vernet. - "The Bath of Venus"; after Raoux. - "Two subjects"; after G. Metsu. - "Jupiter and Calisto"; after N. Poussin. - "St. Margaret"; after Correggio. - "Child playing with Cupid"Deshayes Gendron, oculist"; after Rigaud. - "Jean Baptiste Rousseau"; after Aved. - "Jean Mariette, engraver"; after Pesne.</s><s>Work.:Subjects after 
# various masters. - "The Magdalen"; after Correggio; for the Dresden Gallery. - "Diogenes with his Lantern"; after Spagnoletto; for the same. - "Quos Ego"; after Rubens. - "The Two Sons of Rubens"; after the same; for the Dresden Gallery. - "Neptune appeasing the Tempest"; after the same. - "Charity with Three Children"; after Albani. - "The Triumph of Venus"; after Boucher. - "Les Amusemens de la Campagne"; after Boucher. - "Latona"; after 
# J. Jouvenet. - "Four Marine subjects"; after Joseph Vernet. - "The Bath of Venus"; after Raoux. - "Two subjects"; after G. Metsu. - "Jupiter and Calisto"; after N. Poussin. - "St. Margaret"; after Correggio. - "Child playing with Cupid";'}]
# """
# # Convert single quotes to double quotes and decode
# passages = json.loads(passages_string.replace("'", '"'))

# # Function to get the string label from the predicted class ID
# def get_label_from_id(class_id):
#     labels = model.config.id2label
#     return labels[class_id]

# def evaluate_passage(fact, passage_text):
#     # Clean the passage
#     passage_text = clean_text(passage_text)
        
#     # Tokenize the fact and passage and obtain model inputs
#     inputs = tokenizer(fact, passage_text, return_tensors="pt", truncation=True)

#     # Get model predictions
#     outputs = model(**inputs)

#     # Obtain the predicted class ID and label
#     predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
#     predicted_label = get_label_from_id(predicted_class_id)
        
#     # Calculate the confidence of the prediction
#     probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
#     confidence = probabilities[0][predicted_class_id].item()
        
#     return predicted_label, confidence

# # Evaluate the fact against the group of passages
# def get_final_assessment(fact, passages) -> str:
#     # Clean the fact
#     fact = clean_text(fact)

#     # Confidence threshold
#     confidence_threshold = 0.50  # You can adjust this value based on your requirements
#     supported_by_any_passage = False

#     for passage in passages:
#         predicted_label, confidence = evaluate_passage(fact, passage['text'])
        
#         # Print the result
#         print(f"Fact: '{fact}'\nPassage: '{passage['text']}'\nPrediction: {predicted_label.capitalize()} with confidence {confidence:.2f}\n")
        
#         # Check if the fact is supported by the passage with confidence above the threshold
#         if predicted_label == "entailment" and confidence > confidence_threshold:
#             supported_by_any_passage = True

#     # Print the final assessment
#     assessment = ""
#     if supported_by_any_passage:
#         print("Final Assessment: Supported")
#         assessment = "S"
#     else:
#         print("Final Assessment: Not Supported")
#         assessment = "NS"

#     return assessment

# def get_final_assessment2(fact, passages):
#     # Thresholds for decision-making
#     SUPPORT_THRESHOLD = 0.7
#     CONTRADICTION_THRESHOLD = 0.7

#     support_confidences = []
#     contradiction_confidences = []

#     for passage in passages:
#         # Use the entire passage text for evaluation
#         predicted_label, confidence = evaluate_passage(fact, passage['text'])
        
#         # Print the result
#         print(f"Fact: '{fact}'\nPassage: '{passage['text']}'\nPrediction: {predicted_label.capitalize()} (Confidence: {confidence:.2f})\n")
        
#         # Store confidence scores based on prediction
#         if predicted_label == "entailment":
#             support_confidences.append(confidence)
#         elif predicted_label == "contradiction":
#             contradiction_confidences.append(confidence)

#         # Calculate average confidences
#         avg_support_confidence = sum(support_confidences) / len(support_confidences) if support_confidences else 0
#         avg_contradiction_confidence = sum(contradiction_confidences) / len(contradiction_confidences) if contradiction_confidences else 0

#         print(f"Average Support Confidence: {avg_support_confidence:.2f}")
#         print(f"Average Contradiction Confidence: {avg_contradiction_confidence:.2f}")

#         # Final decision based on average confidences and thresholds
#         if avg_support_confidence > SUPPORT_THRESHOLD and avg_support_confidence > avg_contradiction_confidence:
#             print("Final Assessment: Supported")
#             return "S"
#         else:
#             print("Final Assessment: Not Supported")
#             return "NS"

# assessment = get_final_assessment2(fact, passages)
# print("Assessment: ", assessment)

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
ent_model = EntailmentModel(roberta_ent_model, ent_tokenizer)
fact_checker = EntailmentFactChecker(ent_model)

fact = "Hibo Wardere is a campaigner." 
passage_1 = "Hibo Wardere is a campaigner against female genital mutilation."

print(ent_tokenizer(fact, passage_1, return_tensors='pt', truncation=True, padding=True))

res, conf = ent_model.check_entailment(fact_checker.clean_text(fact), fact_checker.clean_text(passage_1))
# res = fact_checker.predict(fact, passages)

print("res ", res, " ", conf)