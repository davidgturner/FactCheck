import numpy as np

# Sample 3-class probabilities for multiple instances
three_class_probs = np.array([[0.2, 0.5, 0.3],  # Probabilities for instance 1
                              [0.7, 0.2, 0.1],  # Probabilities for instance 2
                              [0.4, 0.4, 0.2]]) # Probabilities for instance 3

# Mapping 3-class probabilities to 2-class probabilities
two_class_probs = np.zeros((three_class_probs.shape[0], 2))  # Initializing a new array for 2-class probabilities

two_class_probs[:, 0] = three_class_probs[:, 0]  # Class X probability is same as Class A probability
two_class_probs[:, 1] = three_class_probs[:, 1] + three_class_probs[:, 2]  # Class Y probability is sum of Class B and Class C probabilities

print("Two class probabilities:")
print(two_class_probs)

# Getting the 2-class predictions
two_class_predictions = np.argmax(two_class_probs, axis=1)

print("\nTwo class predictions:")
print(two_class_predictions)