# Customer Intelligence & Support Copilot — Teaching Kit

> An offline teaching project using fictional Instamart-style customer, order, and support data. It needs no hosted LLM, API key, or network access.

## Learning outcomes

By completing the project, students will learn how to:

1. Read local CSV exports and check for missing, repeated, or invalid values.
2. Clean data and save trusted tables to SQLite.
3. Use SQL and pandas to answer basic business questions.
4. Group customers by shopping behavior and predict possible customer lapse.
5. Remove common personal details from support messages and route tickets safely.
6. Find relevant policy text and use bounded tools to demonstrate a refund workflow.

## Setup and run

Python 3.11+ is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_project.py
```


## Choose a class notebook

There are two notebooks, each focused on one class:

- [Machine learning concepts](./machine_learning_concepts.ipynb): RFM customer groups and lapse prediction.
- [Main tasks](./customer_intelligence_main_task.ipynb): data preparation, ticket triage, policy retrieval, and the simulated refund agent.

Open a notebook from the project root and run its cells from top to bottom. Both notebooks use local synthetic data and work offline.

## Repository map

The project is organized by workflow stage. Start with `run_project.py` to run the stages in sequence, or use the individual scripts and notebooks to explore a particular topic.

- `generate_data.py` — creates the fictional CSV exports and policy text files.
- `mock_llm.py` — local, deterministic stand-in for text classification, policy answers, and agent tool choices.
- `run_project.py` — runs the project stages in order.
- `src/part_a_clean.py` — audits raw data, addresses documented quality issues, and writes clean CSVs and SQLite tables.
- `src/part_a_validate.py` — checks the cleaned data against expected project checkpoints.
- `src/part_a_sql.py` and `sql/queries.sql` — run example SQL and pandas analyses for the data foundation.
- `src/part_b_segments.py` — builds RFM features, compares K-means cluster counts, and saves customer segments.
- `src/part_b_lapse.py` — trains and evaluates customer-lapse models using earlier order history.
- `src/part_c_triage.py` — redacts common personal details, classifies support tickets, and selects a safe route or human review.
- `src/part_d_agent.py` — retrieves policy text and runs bounded refund-agent demonstrations using local tools.
- `data/raw/` — generated CSV inputs; `data/raw_manual/` — independent starter copies for comparison.
- `data/clean/` and `data/instamart.db` — cleaned data used by later stages.
- `data/policies/` — five fictional policy documents used by the policy retrieval exercise.
- `outputs/` — generated analysis and demonstration results; `state/` — saved agent checkpoints for resumable demonstrations.

## How the data flow works

1. `generate_data.py` creates local CSV files. No API is contacted.
2. Part A checks the raw rows, fixes documented data problems, and saves clean CSVs and SQLite tables.
3. Parts B–D use the clean data for customer analysis, support-ticket triage, and policy/refund demonstrations.
4. The scripts save results under `outputs/`. You can inspect these files after the run.

Keep `data/raw_manual/` for manual comparisons: regenerating `data/raw/` does not update those copies.

## Part A — Data foundation

Part A introduces data quality and reproducible preparation. The cleaning script profiles the raw CSV exports, reports issues such as duplicates, missing values, and invalid order details, and applies the project's documented cleaning rules. It writes trusted tables to `data/clean/` and `data/instamart.db`. A validation step checks key expectations, while the SQL and pandas examples answer basic questions about customers, products, and orders.

## Part B — Customer segments and lapse prediction

Part B uses the prepared order history to explore customer behavior. RFM-style features summarize recency, frequency, and monetary value; K-means is used to form example customer groups, and a cluster-count comparison illustrates one way to assess candidate groupings. A separate model predicts which customers may not order in September, using only earlier order history to avoid using future information as a feature. The segments and model outputs are educational examples, not reliable business decisions. Results are saved in `outputs/customer_segments.csv`, `outputs/segment_k_sweep.csv`, `outputs/lapse_model.joblib`, and `outputs/retention_list.csv`.

## Part C — Ticket triage and privacy

Part C demonstrates basic privacy-aware support routing. The script removes common email, phone, and order-ID patterns before classifying ticket text, then routes tickets to a suitable support queue or sends uncertain cases for human review. Fraud-related cases are directed to the security team. The redaction patterns can miss sensitive details, so this is not a substitute for production privacy controls; the classifier is also a local simulation. Results are saved in `outputs/triage_results.csv` and related evaluation files.

## Part D — Policy retrieval and refund tools

Part D combines local policy retrieval with a bounded, simulated tool workflow. The policy assistant searches the fictional policy files for relevant passages. The refund agent checks the order and applicable rules before taking an allowed action or escalating the case: in this teaching scenario, completed orders delivered more than 45 minutes late qualify for a full refund, while refund amounts over INR 1,000 are escalated. The agent's progress is saved in `state/` so selected demonstrations can resume. All actions are simulated; no real customer account or money is affected. Evaluation results are in `outputs/policy_eval_results.csv` and `outputs/policy_eval_zero_overlap_results.csv`.

## GitHub suggestions

Use these official resources to explore the tools and concepts used in this project:
- [pandas documentation](https://pandas.pydata.org/docs/)
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
- [Python tutorial](https://docs.python.org/3/tutorial/)
- [Jupyter documentation](https://docs.jupyter.org/)
- [SQLite documentation](https://www.sqlite.org/docs.html)

## Limitations

- All records and policies are fictional and are not official Instamart policies.
- `mock_llm.py` is a deterministic teaching simulation, not a hosted language model.
- The policy retriever can miss synonyms and paraphrases.
- The lapse model and retention list are educational examples, not production predictions or causal evidence.
- A production refund system would need authentication, authorization, privacy controls, stronger audit records, and human approval.
