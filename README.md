# Scalable Association Rule Mining and Estimator

---

## 1. Project Title
**Scalable Association Rule Mining and Estimator**

---

## 2. Problem Statement
Traditional market basket analysis and association rule mining algorithms (such as standard Apriori) rely heavily on static user-defined hyperparameter thresholds (e.g., fixed minimum support and confidence). Setting these parameters manually often results in either a rule explosion (thousands of redundant/trivial rules) or rule sparsity (missing subtle, highly actionable cross-selling opportunities). Furthermore, static rule generation lacks automated validation on unseen transactions and direct integration with executive decision-support systems.

---

## 3. Objective 1 Scope
This repository implements **Objective 1** of the capstone project:
> *"Develop and validate the basic transaction-mining pipeline using the selected Groceries dataset and establish Apriori as the baseline for association-rule mining."*

*Note: Advanced modules such as FP-Growth, adaptive parameter recommendation, rule redundancy reduction, AI-based business analysis, business recommendations, AI Copilot, and automated report generation will be integrated in subsequent project phases.*

---

## 4. Dataset Overview
We use the benchmark **Groceries Dataset** (`Groceries_dataset.csv`).

### Dataset Summary Statistics (Calculated Programmatically)
- **Raw CSV Records**: 38,765 purchase records
- **Raw CSV Columns**: 3 (`Member_number`, `Date`, `itemDescription`)
- **Missing Values**: 0 missing values
- **Exact Duplicate Rows**: 759 duplicate records
- **Clean Records**: 38,006 unique item-level records
- **Unique Customers**: 3,898 customers (`Member_number`)
- **Unique Products**: 167 product categories (`itemDescription`)
- **Unique Purchase Dates**: 728 dates (01-Jan-2014 to 30-Dec-2015)

---

## 5. Dataset Structure
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Member_number` | Integer / String | Unique identifier assigned to each customer |
| `Date` | String | Date of purchase (DD-MM-YYYY format) |
| `itemDescription` | String | Description of the purchased grocery item |

---

## 6. Transaction Definition
The original dataset does not contain an explicit Transaction ID column. Each row represents a single item purchase by a member on a given date.

Therefore, a unique transaction (basket) is defined as:
$$\text{Transaction\_ID} = \text{Member\_number} + \text{"\_"} + \text{Date}$$

Using this formulation:
- **Total Unique Transactions / Baskets**: **14,963**
- **Average Basket Size**: **~2.54 items** per transaction (exact: 2.539998)

---

## 7. Data Preprocessing Workflow
1. **CSV Ingestion & Column Verification**: Verify required columns (`Member_number`, `Date`, `itemDescription`).
2. **Deduplication**: Remove 759 exact duplicate rows.
3. **Missing Value Handling**: Check and handle any missing or null values.
4. **Transaction ID Generation**: Create unique `Transaction_ID` for each customer-date combination.
5. **Basket Aggregation & Inner Deduplication**: Group items by `Transaction_ID` into sets to ensure no item appears more than once within the same basket.

---

## 8. One-Hot Encoding
To prepare transactions for Apriori mining, the variable-length list of item baskets is transformed into a binary transaction-item matrix using `mlxtend.preprocessing.TransactionEncoder`.
- **Rows ($N$)**: Transactions (14,963 total)
- **Columns ($M$)**: 167 unique grocery products
- **Values**: `True` (1) if the item was purchased in that transaction, `False` (0) otherwise.

---

## 9. Development vs. Validation Split
The transaction-level dataset is split into an 80/20 train/test style split at the **TRANSACTION LEVEL** using a reproducible random seed (`random_state=42`):
- **Development / Mining Set (80%)**: **11,970 transactions**
- **Validation Set (20%)**: **2,993 transactions**

> [!IMPORTANT]
> **Supervised ML vs. Pattern Mining Validation**:
> Association rule mining is an unsupervised pattern discovery method. The Development set is used to discover frequent itemsets and generate rules. The separate 20% Validation set evaluates whether the discovered item relationships hold consistently on unseen customer transactions.

---

## 10. Apriori Baseline
Apriori is established as the foundational benchmark algorithm:
1. **Frequent Itemset Mining**: Discovers candidate itemsets $I$ satisfying $Support(I) \ge \text{min\_support}$ on the Development Set.
2. **Default Baseline Thresholds**:
   - `min_support` = 0.001 (0.1% of development transactions)
   - `min_confidence` = 0.05 (5% confidence threshold)

*Note: Default values are customizable in the user interface and are not claimed as scientifically optimal.*

---

## 11. Association Rule Generation
Rules are generated in the standard antecedent-to-consequent format:
$$X \to Y \quad \text{where } X \cap Y = \emptyset$$

Each generated rule includes:
- **Antecedents ($X$)**: The itemset purchased first.
- **Consequents ($Y$)**: The itemset purchased in conjunction with $X$.

---

## 12. Rule Evaluation Metrics
For every rule $X \to Y$, the following standard association metrics are calculated:

1. **Support**:
   $$Support(X \to Y) = \frac{\text{Count}(X \cup Y)}{N}$$
2. **Confidence**:
   $$Confidence(X \to Y) = \frac{Support(X \cup Y)}{Support(X)}$$
3. **Lift**:
   $$Lift(X \to Y) = \frac{Confidence(X \to Y)}{Support(Y)} = \frac{Support(X \cup Y)}{Support(X) \times Support(Y)}$$
4. **Leverage**:
   $$Leverage(X \to Y) = Support(X \cup Y) - [Support(X) \times Support(Y)]$$

---

## 13. Validation Methodology vs. Unit Testing vs. Baseline Evaluation
It is crucial to distinguish between these three distinct evaluation pillars in capstone presentations:

- **Unit Testing**: Verifies system software integrity using small, manually verifiable test datasets to ensure functions calculate metrics correctly (Pytest suite: UT01–UT13).
- **Rule Validation**: Measures generalization performance by evaluating rules mined from the Development set against the unseen 20% Validation set ($Support_{val}$, $Confidence_{val}$, $Lift_{val}$, $Leverage_{val}$).
- **Baseline Evaluation**: Benchmarks system performance metrics (execution time, RSS memory footprint, total itemsets, total rules) under standard hyperparameter configurations.

---

## 14. Unit Testing Suite (UT01 – UT13)
The project includes a full unit test suite implemented with `pytest`:

| Test ID | Description | Module Tested |
| :--- | :--- | :--- |
| **UT01** | Dataset loading with valid CSV | `src.preprocessing.load_dataset` |
| **UT02** | Required column validation | `src.preprocessing.verify_columns` |
| **UT03** | Exact duplicate removal | `src.preprocessing.clean_dataset` |
| **UT04** | Transaction ID creation (`Member_number` + `Date`) | `src.preprocessing.create_transaction_id` |
| **UT05** | Transaction basket creation | `src.preprocessing.create_baskets` |
| **UT06** | Repeated item removal within same basket | `src.preprocessing.create_baskets` |
| **UT07** | One-hot binary matrix encoding | `src.encoder.encode_transactions` |
| **UT08** | Apriori execution on known test dataset | `src.mining.run_apriori` |
| **UT09** | Association rule generation | `src.mining.generate_rules` |
| **UT10** | Support metric calculation verification | `src.mining.run_apriori` |
| **UT11** | Confidence metric calculation verification | `src.mining.generate_rules` |
| **UT12** | Lift metric calculation verification | `src.mining.generate_rules` |
| **UT13** | Validation metric evaluation on unseen data | `src.validation.evaluate_rules_on_validation` |

---

## 15. Installation & Setup

### Prerequisites
- Python 3.9+ installed.

### Installation Steps
```bash
# 1. Clone or navigate to project workspace
cd "d:\capstone review 2"

# 2. Install required dependencies
pip install -r requirements.txt
```

---

## 16. How to Run the Application & Tests

### Option A: Run Interactive Streamlit Web App
```bash
streamlit run app.py
```
This launches a browser interface featuring:
- Dataset Overview & Interactive Preview
- Preprocessing & Basket Stats
- 80/20 Transaction Split Controls
- Configurable Support & Confidence Sliders
- Apriori Mining Execution & Rule Tables
- Side-by-Side Development vs. Validation Metrics
- Live Pytest Test Execution Button
- System Resource & Performance Benchmark Logs

### Option B: Run Unit Tests
```bash
pytest tests/test_pipeline.py -v
```

---

## 17. Project Structure
```
d:\capstone review 2\
├── data/
│   └── Groceries_dataset.csv       # Benchmark Groceries dataset CSV
├── src/
│   ├── __init__.py                 # Python package init
│   ├── preprocessing.py            # Data loading, column verification, deduplication, transaction ID & basket creation
│   ├── encoder.py                  # TransactionEncoder one-hot matrix transformation
│   ├── split.py                    # 80/20 transaction-level split
│   ├── mining.py                   # Apriori execution, frequent itemsets, association rule generation
│   ├── validation.py               # Evaluation of rules on unseen validation dataset
│   └── metrics.py                  # Benchmarking execution time and psutil memory tracking
├── tests/
│   ├── __init__.py                 # Test package init
│   └── test_pipeline.py            # Automated Pytest suite (UT01 to UT13)
├── results/
│   └── baseline_metrics.json       # Programmatically exported baseline metrics
├── app.py                          # Streamlit interactive UI application
├── requirements.txt                # Dependencies list
└── README.md                       # Complete student presentation & technical reference guide
```

---

## 18. Current Limitations (Objective 1)
1. **Static Threshold Sensitivity**: Standard Apriori requires manual trial-and-error setting of minimum support and confidence.
2. **Computational Overhead**: Apriori generates large numbers of candidate itemsets compared to tree-based structures.
3. **No Rule Pruning**: Redundant rules (e.g. sub-rules with identical confidence) are not yet automatically filtered.
4. **No Semantic AI Layer**: Business interpretation of rules requires manual expert review at this stage.

---

## 19. Future Roadmap & Upcoming Modules
- **Module 2**: FP-Growth algorithm implementation for scalable pattern mining.
- **Module 3**: Adaptive hyperparameter recommendation engine.
- **Module 4**: Rule redundancy reduction & pruning algorithms.
- **Module 5**: AI-powered business insight generation & executive recommendation copilot.
- **Module 6**: Interactive visual graph analytics and automated report exporter.
