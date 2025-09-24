
# Bank Customer Segmentation

This project applies **clustering and machine learning techniques** to segment bank clients based on their financial behavior, product usage, and relationship with the bank.  

### Objectives
- Identify homogeneous groups of clients with similar characteristics.
- Analyze product usage patterns (savings, credit, payroll, loans, etc.).
- Provide insights for marketing strategies, customer retention, and risk management.

### Methodology
- **Data preprocessing**: cleaning, feature engineering (lags, deltas, ratios, linear regressions, max/min indicators).
- **Clustering models**: K-Means, hierarchical clustering, PAM/K-Medoids.
- **Random Forest distance calculation**: building a similarity matrix from tree co-occurrences to better capture non-linear relationships between clients before clustering.
- **Evaluation metrics**: Adjusted Rand Index (ARI), silhouette score, Van Dongen metric.
- **Visualization**: embeddings with UMAP, distribution of features across clusters.

### Results
The segmentation revealed distinct client profiles:
- Active salaried clients.
- High product usage and high profitability clients.
- Low activity and negative balance clients.
- Others with specific transactional or product patterns.

These insights can be used for **customer targeting, loyalty campaigns, and risk assessment**.

---
