import json
import re
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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


fact = "Jean Daullé was born in 1703." 

passages_string = """
[{'title': 'Jean Daullé', 'text': '<s>Jean Daullé (18 May 1703 – 23 April 1763) was a French engraver.</s><s>Biography. He was the son of Jean Daullé, a silversmith, and his wife, Anne née Dennel. At the age of fourteen, he received training from an engraver named Robart, at the priory of Saint-Pierre d'Abbeville. He then went to Paris, and worked at the studios of Robert Hecquet (1693-1775), who was also originally from Picardy. In 1735, his work attracted the attention of the engraver and merchant, Pierre-Jean Mariette, who provided him with professional recommendations. Soon after, he was approached by the painter, Hyacinthe Rigaud, who wanted to 
make him his official engraver. In 1742, Daullé was received at the Académie Royale de Peinture et de Sculpture, with his presentation, "Hyacinthe Rigaud Painting his Wife", after a work by Rigaud. He was also admitted as a member of the academy in Augsbourg. Eventually named "Engraver to the King",'}, {'title': 'Jean Daullé', 'text': ' he trained the future publisher and print dealer, Pierre-François Basan, as well as the German engraver Jean-Georges Wille. Around 1745, he married Gabrielle-Anne Landry and they had five children. Overwhelmed by his large family, his work suffered.</s><s>Work. He engraved several portraits and plates of historical and other subjects, which are chiefly executed with the graver in a clear and firm style, which entitles him to rank with the ablest artists of his time. He marked his works J. D. The following are his principal plates:</s><s>Work.:Portraits. - "Catherine, Countess of Feuquières, daughter of Pierre Mignard"; after Mignard. - "Hyacinthe Rigaud, painter"; after Rigaud; engraved for his reception at the Academy in 1742. - "Marguerite de Valois, 
Countess of Caylus"; after the same. - "Charles Edward Stuart, son of the Pretender". - "Clementina, Princess of Poland, his consort"; after David. 
- "Madame Favart, in the part of 'Bastienne;' " after Carle van Loo.'}, {'title': 'Jean Daullé', 'text': " after Van Dyck. A detailed account of this artist's works is contained in Delignière's 'Catalogue raisonné de l'oeuvre gravé de d' Abbeville,' 1872, 8vo.</s><s>References. - - Attribution:</s>"}, {'title': 'Jean Daullé', 'text': ' - "Claude Deshayes Gendron, oculist"; after Rigaud. - "Jean Baptiste Rousseau"; after Aved. - "Jean Mariette, engraver"; after Pesne.</s><s>Work.:Subjects after 
various masters. - "The Magdalen"; after Correggio; for the Dresden Gallery. - "Diogenes with his Lantern"; after Spagnoletto; for the same. - "Quos Ego"; after Rubens. - "The Two Sons of Rubens"; after the same; for the Dresden Gallery. - "Neptune appeasing the Tempest"; after the same. - "Charity with Three Children"; after Albani. - "The Triumph of Venus"; after Boucher. - "Les Amusemens de la Campagne"; after Boucher. - "Latona"; after 
J. Jouvenet. - "Four Marine subjects"; after Joseph Vernet. - "The Bath of Venus"; after Raoux. - "Two subjects"; after G. Metsu. - "Jupiter and Calisto"; after N. Poussin. - "St. Margaret"; after Correggio. - "Child playing with Cupid";'}]
"""

# Extract titles and texts using regex
titles = re.findall(r"'title':\s*'([^']+)'", passages_string)
texts = re.findall(r"'text':\s*'([^']+)'", passages_string)

# Build a list of dictionaries
passages = [{"title": title, "text": text} for title, text in zip(titles, texts)]

# Convert to JSON
passages_json = json.dumps(passages, indent=4)

print(passages_json)

# print(passages)

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


# fact = ""

# passages = 

# Function to get the string label from the predicted class ID
def get_label_from_id(class_id):
    labels = model.config.id2label
    return labels[class_id]

# Clean the fact
fact = clean_text(fact)

# Evaluate the fact against the group of passages
support_count = 0
total_passages = len(passages)

def evaluate_passage(fact, passage_text):
    # Clean the passage
    passage_text = clean_text(passage_text)
    
    # Tokenize the fact and passage and obtain model inputs
    inputs = tokenizer(fact, passage_text, return_tensors="pt", truncation=True, max_length=512)

    # Get model predictions
    outputs = model(**inputs)

    # Obtain the predicted class ID and label
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = get_label_from_id(predicted_class_id)
    
    return predicted_label

for passage in passages:
    # Use the entire passage text for evaluation
    predicted_label = evaluate_passage(fact, passage['text'])
    
    # Print the result
    print(f"Fact: '{fact}'\nPassage: '{passage['text']}'\nPrediction: {predicted_label.capitalize()}\n")
    
    # Count the number of supports (entailments)
    if predicted_label == "entailment":
        support_count += 1

# Print the final binary assessment based on majority vote
if support_count > total_passages / 2 or (support_count == total_passages / 2 and total_passages % 2 == 0):
    print("Final Assessment: Supported")
else:
    print("Final Assessment: Not Supported")