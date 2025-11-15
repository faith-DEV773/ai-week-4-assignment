2. HOW IBM AI FAIRNESS 360 ADDRESSES THESE BIASES

What is AI Fairness 360?
IBM's AI Fairness 360 (AIF360) is an open-source toolkit with 70+ fairness metrics and 10+ bias mitigation algorithms.
GitHub: https://github.com/Trusted-AI/AIF360

A. Bias Detection Capabilities
1. Disparate Impact Analysis
pythonfrom aif360.metrics import BinaryLabelDatasetMetric

# Measures if outcomes differ between groups
metric = BinaryLabelDatasetMetric(dataset, 
    unprivileged_groups=[{'team': 'junior'}],
    privileged_groups=[{'team': 'senior'}])

disparate_impact = metric.disparate_impact()
# Values < 0.8 indicate bias (EEOC four-fifths rule)

print(f"Disparate Impact: {disparate_impact:.2f}")
# Output: 0.65 → Junior team issues 35% less likely to be high priority
Application to Our Model:

Detect if certain teams' issues systematically deprioritized
Identify protected group disparities
Quantify bias magnitude

2. Statistical Parity Difference
pythonstat_parity = metric.statistical_parity_difference()
# Measures difference in positive outcome rates between groups

# Target: Close to 0 (equal treatment)
# Values > 0.1 or < -0.1 indicate bias
Interpretation for Issue Priority:
Team A High Priority Rate: 60%
Team B High Priority Rate: 35%
Statistical Parity Difference: 0.25

Conclusion: Team A's issues 25% more likely to be high priority → BIAS DETECTED
3. Equal Opportunity Difference
pythoneq_opp_diff = metric.equal_opportunity_difference()
# Measures if truly important issues treated equally across groups
Critical for Software Engineering:
Ensures genuinely critical bugs get high priority regardless of who reported them.

B. Bias Mitigation Algorithms
IBM AIF360 provides three mitigation approaches:
1. Pre-Processing: Fix the Data Before Training
Reweighing Algorithm
pythonfrom aif360.algorithms.preprocessing import Reweighing

# Adjusts training sample weights to achieve fairness
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

dataset_transf = RW.fit_transform(dataset)
How It Works:

Underrepresented groups get higher sample weights
Overrepresented groups get lower weights
Model sees "balanced" distribution during training

Application to Issue Priority:
Original Distribution:
- Senior Team issues: 1000 samples (weight: 1.0)
- Junior Team issues: 200 samples (weight: 1.0)

After Reweighing:
- Senior Team issues: 1000 samples (weight: 0.6)
- Junior Team issues: 200 samples (weight: 3.0)

Effective Distribution:
- Senior Team: 600 effective samples
- Junior Team: 600 effective samples
→ Balanced training
Benefit: Model learns equal importance of both teams' issues.

Disparate Impact Remover
pythonfrom aif360.algorithms.preprocessing import DisparateImpactRemover

DIR = DisparateImpactRemover(repair_level=0.8)
dataset_transf = DIR.fit_transform(dataset)
How It Works:

Removes correlation between features and protected attributes
Preserves predictive power while reducing bias

Example:
Original: Developer experience strongly correlates with age
After DIR: Experience adjusted to remove age correlation
Result: Model can't discriminate based on age

2. In-Processing: Fairness During Training
Adversarial Debiasing
pythonfrom aif360.algorithms.inprocessing import AdversarialDebiasing

# Uses two networks:
# 1. Predictor: Tries to predict issue priority
# 2. Adversary: Tries to predict protected attribute (e.g., team)

# Predictor trained to fool adversary
# Result: Predictions independent of protected attribute

debiased_model = AdversarialDebiasing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups,
    scope_name='debiased_classifier')

debiased_model.fit(dataset_train)
How It Works:
┌─────────────┐
│   Features  │
└──────┬──────┘
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
┌─────────────┐         ┌──────────────┐
│  Predictor  │◄────────│  Adversary   │
│ (Priority)  │ Fights  │ (Team ID)    │
└─────────────┘         └──────────────┘

Goal: Predictor learns to predict priority WITHOUT revealing team
Benefit: Model physically cannot use protected attributes for predictions.

Prejudice Remover
pythonfrom aif360.algorithms.inprocessing import PrejudiceRemover

# Adds fairness regularization term to loss function
# Loss = Prediction Error + λ × Fairness Penalty

PR = PrejudiceRemover(eta=1.0)  # eta controls fairness emphasis
PR.fit(dataset_train)
Mathematical Approach:
Traditional ML: Minimize prediction error only
Fairness-Aware ML: Minimize (prediction error + fairness violation)

Result: Slight accuracy trade-off for significant fairness gain
Typical Trade-off:

Accuracy: 92% → 89% (3% drop)
Fairness: Disparate Impact 0.65 → 0.95 (46% improvement)

Business Decision: Is 3% accuracy worth eliminating discrimination? Usually YES.

3. Post-Processing: Adjust Outputs After Training
Calibrated Equalized Odds
pythonfrom aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

# Adjusts prediction thresholds differently for each group
# Ensures equal true positive and false positive rates

cpp = CalibratedEqOddsPostprocessing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

cpp.fit(dataset_val, dataset_val_pred)
dataset_test_pred = cpp.predict(dataset_test_pred)
How It Works:
Group A: Threshold = 0.6 for "High Priority"
Group B: Threshold = 0.4 for "High Priority"

Effect: Group B (historically disadvantaged) gets lower threshold
Result: Equal treatment in final predictions
Example in Issue Priority:
Junior Team Issue:
- Model Score: 0.55
- Threshold: 0.4 (adjusted)
- Prediction: High Priority ✓

Senior Team Issue:
- Model Score: 0.55
- Threshold: 0.6 (adjusted)
- Prediction: Medium Priority ✓

Both with same score, but adjustments ensure fairness

C. Monitoring and Continuous Auditing
Fairness Dashboard
pythonfrom aif360.metrics import ClassificationMetric

# Monitor fairness metrics in production
metric = ClassificationMetric(
    dataset_true, dataset_pred,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups)

# Key metrics to monitor weekly:
metrics = {
    'Disparate Impact': metric.disparate_impact(),
    'Statistical Parity': metric.statistical_parity_difference(),
    'Equal Opportunity': metric.equal_opportunity_difference(),
    'Theil Index': metric.theil_index(),  # Overall inequality
}

# Alert if any metric exceeds threshold
for metric_name, value in metrics.items():
    if abs(value - FAIR_VALUE) > THRESHOLD:
        send_alert(f"BIAS DETECTED: {metric_name} = {value}")
Automated Bias Monitoring System:
Daily: Calculate fairness metrics on production predictions
Weekly: Generate fairness report for stakeholders
Monthly: Retrain model with bias mitigation if drift detected
Quarterly: Full audit with external review
