import itertools
import numpy as np

# Define HMM parameters
hidden_states = ["Sunny", "Rainy"]
observations = ["Walk", "Shop", "Clean"]

# Initial probabilities (π)
initial_probs = {"Sunny": 0.6, "Rainy": 0.4}

# Transition probabilities (P)
transition_probs = {
    "Sunny": {"Sunny": 0.7, "Rainy": 0.3},
    "Rainy": {"Sunny": 0.4, "Rainy": 0.6},
}

# Emission probabilities (B)
emission_probs = {
    "Sunny": {"Walk": 0.6, "Shop": 0.3, "Clean": 0.1},
    "Rainy": {"Walk": 0.1, "Shop": 0.4, "Clean": 0.5},
}

# Observation sequence
observation_sequence = ["Walk", "Walk", "Shop"]

# Generate all possible hidden state sequences
num_obs = len(observation_sequence)
all_hidden_sequences = list(itertools.product(hidden_states, repeat=num_obs))

# Function to calculate joint probability P(X, Y)
def calculate_joint_probability(hidden_sequence, observation_sequence):
    joint_prob = initial_probs[hidden_sequence[0]]  # Start with P(X1)
    joint_prob *= emission_probs[hidden_sequence[0]][observation_sequence[0]]  # P(Y1 | X1)
    
    # Iterate through the sequence
    for t in range(1, len(hidden_sequence)):
        joint_prob *= transition_probs[hidden_sequence[t - 1]][hidden_sequence[t]]  # P(Xt | Xt-1)
        joint_prob *= emission_probs[hidden_sequence[t]][observation_sequence[t]]  # P(Yt | Xt)
    
    return joint_prob

# Calculate joint probabilities for all hidden state sequences
results = []
for hidden_sequence in all_hidden_sequences:
    joint_prob = calculate_joint_probability(hidden_sequence, observation_sequence)
    results.append((hidden_sequence, joint_prob))

# Find the most likely hidden sequence
most_likely_sequence = max(results, key=lambda x: x[1])

# Display results
print(f"Observation Sequence: {observation_sequence}\n")
print("All Possible Hidden State Sequences and Their Joint Probabilities:")
for seq, prob in results:
    print(f"Hidden Sequence: {seq}, Joint Probability: {prob:.6f}")

print("\nMost Likely Hidden State Sequence:")
print(f"Hidden Sequence: {most_likely_sequence[0]}, Joint Probability: {most_likely_sequence[1]:.6f}")