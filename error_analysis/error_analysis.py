
import pandas as pd

# Load the data from the provided CSV file
data = pd.read_csv('/mnt/data/results_comp.csv')

# Display the first few rows to understand the structure
data.head()





# Extracting False Positives
false_positives = data[(data['Prediction Label'] == 'S') & (data['Golden Label'] == 'NS')].sample(n=10, random_state=42)

# Extracting False Negatives
false_negatives = data[(data['Prediction Label'] == 'NS') & (data['Golden Label'] == 'S')].sample(n=10, random_state=42)

false_positives, false_negatives




# Function to categorize errors based on the provided criteria
def categorize_error(fact):
    categories = {
        "Ambiguity": [],
        "Contextual Information Needed": [],
        "Specificity": [],
        "Temporal Information": [],
        "Omission or Inaccuracies": [],
        "Language or Syntax Complexity": []
    }
    
    # This is a simplistic approach. A more rigorous analysis would require domain knowledge
    # and a deeper understanding of the data sources and the model's training data.
    
    # Check for ambiguity
    if "?" in fact or fact.startswith("Either") or "and/or" in fact:
        categories["Ambiguity"].append(fact)
    
    # Check for potential need for contextual information
    if "known for" in fact or "famous for" in fact:
        categories["Contextual Information Needed"].append(fact)
    
    # Check for specificity
    if "all" in fact or "every" in fact or "only" in fact:
        categories["Specificity"].append(fact)
    
    # Check for temporal information
    if "since" in fact or "until" in fact or "before" in fact or "after" in fact:
        categories["Temporal Information"].append(fact)
    
    # Check for potential omissions or inaccuracies (hard to determine without deeper context)
    # Placeholder for future checks
    
    # Check for language or syntax complexity (long sentences or complex structures)
    if len(fact.split()) > 12:
        categories["Language or Syntax Complexity"].append(fact)
    
    return {key: value for key, value in categories.items() if value}

# Categorize the errors for false positives
false_positive_categories = {}
for fact in false_positives['Fact']:
    false_positive_categories[fact] = categorize_error(fact)

false_positive_categories



# Categorize the errors for false negatives
false_negative_categories = {}
for fact in false_negatives['Fact']:
    false_negative_categories[fact] = categorize_error(fact)

false_negative_categories




# Extracting the relevant columns for the selected facts
selected_facts_data = data[data['Fact'].isin([
    'Jonathan Haagensenis an actor.',
    'Continuum Analytics is an organization.',
    "'Kill Me, Heal Me.' is popular."
])]

# Displaying the relevant data for these facts
selected_facts_data[['Fact', '# of Total Passages', '# of Passage with S', '# of Passage with NS', 'Golden Label', 'Prediction Label']]


