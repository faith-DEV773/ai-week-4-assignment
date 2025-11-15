Q2: Supervised vs. Unsupervised Learning in Automated Bug Detection
Comparative Analysis Framework

1. SUPERVISED LEARNING for Bug Detection
Definition
A machine learning approach where the model learns from labeled training data where bugs are explicitly marked as "buggy" or "clean."

How It Works in Bug Detection


Training Phase:
Input: Code files with metrics (complexity, lines of code, code churn)
Labels: {Bug: 1, No Bug: 0}
Algorithm: Learns patterns that distinguish buggy from clean code


Prediction Phase:
New Code → Extract Features → Model Predicts → Bug/No Bug


2. UNSUPERVISED LEARNING for Bug Detection
Definition
Machine learning approach that discovers hidden patterns and anomalies in code without labeled data.

How It Works in Bug Detection

Training Phase:
Input: Code features (unlabeled)
Algorithm: Identifies clusters or anomalies
Output: "Normal" code patterns vs. "Unusual" code patterns

Detection Phase:
New Code → Compare to Normal Patterns → Anomaly Score → Flag Suspicious Code
