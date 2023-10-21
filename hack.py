# import nltk
# from typing import List, Dict
# import re
# import string
import nltk
from nltk.corpus import stopwords
from typing import List, Dict
import re
import string

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text: str) -> str:
    """Clean the text by removing punctuation, converting to lowercase, and removing stopwords."""
    text = text.lower()  # convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # remove stopwords
    return text

def jaccard_similarity(str1: str, str2: str) -> float:
    """Compute the Jaccard similarity between two strings."""
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# Sample Wikipedia data: a list of dictionaries with "title" and "text" keys
wiki_data = [
    {
        "title": "Python (programming language)",
        "text": "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant indentation."
    },
    {
        "title": "JavaScript",
        "text": "JavaScript (/ˈdʒɑːvəˌskrɪpt/), often abbreviated as JS, is a programming language that conforms to the ECMAScript specification. JavaScript is high-level, often just-in-time compiled, and multi-paradigm."
    }
]

# A given fact that we want to compare against the Wikipedia data
given_fact = "Python is a popular programming language."

def process_fact(given_fact, passages):
    # Clean the given fact
    cleaned_fact = clean_text(given_fact)

    # Initialize variables to store the max similarity and the corresponding sentence
    max_similarity = 0
    most_similar_sentence = ""

    # Loop over the Wikipedia data
    for passage in wiki_data:
        # Clean and split the text into sentences
        cleaned_text = clean_text(passage["text"])
        sentences = nltk.sent_tokenize(cleaned_text)

        # Loop over the sentences
        for sentence in sentences:
            # Compute the similarity between the sentence and the given fact
            similarity = jaccard_similarity(sentence, cleaned_fact)

            # Update the max similarity and the most similar sentence
            if similarity > max_similarity:
                print("sim sim ", similarity, " for sent ", sentence)
                max_similarity = similarity
                most_similar_sentence = sentence

    return most_similar_sentence, max_similarity

sim_sent, max_sim = process_fact(given_fact, wiki_data)

print("most sim_sent ", sim_sent)
print("max_sim ", max_sim)
