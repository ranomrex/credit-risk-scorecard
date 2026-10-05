# Credit Risk Scorecard and Approval Strategy

Predicting which borrowers will seriously default within 2 years, turning the model into a bank-style points scorecard, and using it to choose the most profitable approval policy.

## Data
Give Me Some Credit (Kaggle): 150,000 borrowers, 10 features, 6.7% default rate. To run the notebook, download `cs-training.csv` from Kaggle and place it in Google Drive under My Drive.

## Approach
1. **Exploration:** found default falls with age (12% under 30, 2% over 70) and jumps with past delinquency (5% with no 90-day lates, 34% with one).
2. **Cleaning:** removed invalid ages, flagged error codes (96/98) and missing income instead of deleting rows, capped outliers at the 99th percentile, log-transformed skewed variables.
3. **Model:** logistic regression on a stratified 70/30 split with standardized features.
4. **Scorecard:** Weight of Evidence binning and Information Value, scaled to points.
5. **Strategy:** simulated approval cutoffs and profit under stated assumptions.

## Results: Raw-Feature Model
| Metric | Train | Test |
|---|---|---|
| AUC | 0.850 | 0.852 |
| Gini | | 0.705 |
| KS | | 0.545 |

Train and test AUC match closely, so the model generalizes well.

![ROC curve](download.png)

**Top risk drivers:** credit utilization, past 30/60/90-day delinquencies. Age lowers risk.

## Points Scorecard (Weight of Evidence)
Each feature was split into groups (quintiles from training data only; late payments as 0, 1, 2+) and each group was given a Weight of Evidence. Information Value ranked the features: utilisation (1.06), 90-day lates (0.84), 30-59 day lates (0.68), 60-89 day lates (0.56), age (0.23); the rest were weak but above 0.02.

Logistic regression on WoE values was scaled to points: 600 points at 50:1 good-to-bad odds, and 20 points doubles the odds.

| Metric | Raw-feature model | WoE scorecard |
|---|---|---|
| Test AUC | 0.852 | 0.854 |
| KS | 0.545 | 0.560 |

All scorecard coefficients point the expected way, which fixes the DebtRatio sign problem in the raw model.

| Score band | Default rate |
|---|---|
| Under 500 | 52% |
| 500 to 550 | 21% |
| 550 to 600 | 4.7% |
| 600 and above | under 1% |

## Approval Strategy
Assumptions: Rs 5,000 profit per good customer, Rs 50,000 loss per default.

| Approval rate | Bad rate | Defaulters rejected | Profit (Rs cr) |
|---|---|---|---|
| 70% | 1.97% | 79% | 12.33 |
| **80%** | **2.53%** | **70%** | **13.00** |
| 90% | 3.45% | 54% | 12.57 |
| 100% | 6.68% | 0% | 5.96 |

**Recommendation:** approve the safest 80%. Profit more than doubles versus approving everyone, and the bad rate falls 62%. An 85% cutoff gives near-equal profit if growth is a priority.

## Limitations and Next Steps
- DebtRatio is unreliable when income is missing, which made its raw-model coefficient counterintuitive (the WoE scorecard corrects this).
- The first two utilisation groups are not in risk order; merging them would make the scorecard fully monotonic.
- Next: compare with gradient boosting.
