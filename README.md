# PAYGUARD — AI PAYMENT RISK & FRAUD AGENT

## Day 1 Execution Report

### Files Created/Modified
- `requirements.txt`: Python dependencies.
- `.gitignore`: Standard python ignores.
- `backend/data/generate_dataset.py`: Script to generate a realistic synthetic transaction dataset.
- `backend/evaluation/baseline_model.py`: Script to train a baseline scikit-learn Random Forest model.
- `tests/test_data_quality.py`: Pytest suite to validate the synthetic dataset structure.
- `DATASET.md`: Documentation for the generated features and dataset structure.
- `README.md`: This file.

### Commands Used
1. `New-Item -ItemType Directory -Force -Path backend\data, backend\models, backend\features, backend\agents, backend\services, backend\evaluation, tests, notebooks`
2. `pip install -r requirements.txt`
3. `python backend/data/generate_dataset.py`
4. `pytest tests/test_data_quality.py`
5. `python backend/evaluation/baseline_model.py`

### Dataset Statistics
- **Total Dataset Size:** 10,000 transactions
- **Fraud Percentage:** 5.0% (500 fraudulent transactions)
- **Train Size:** 7,000 (chronological 0-70%)
- **Validation Size:** 1,500 (chronological 70-85%)
- **Test Size:** 1,500 (chronological 85-100%)

### Baseline Model Performance (Random Forest)
*Performance metrics calculated on the held-out Test Set.*
- **Baseline Precision:** 0.9859
- **Baseline Recall:** 0.8974
- **Baseline F1 Score:** 0.9396
- **False Positive Rate:** 0.0007

### Limitations and Assumptions
- The baseline model primarily utilizes numerical/categorical metadata (e.g. amount, behavioral flags) while stripping direct identifiers to avoid overfitting. 
- Synthetic noise was introduced so that fraudulent transactions have some overlap with legitimate ones (preventing trivial 100% separability).
- IP addresses, device IDs, and customer IDs are currently not feature-engineered (e.g., entity embeddings) in the baseline but serve as grounds for graph/agent-based risk signals later.

### What Remains for Day 2
- Construct explicit rule-based systems to capture specific fraud topologies before ML inference.
- Begin the Multi-Agent setup using a framework like LangChain or specialized LLM agents for interpreting risk signals.
- Set up FastAPI endpoints to score incoming transactions in real-time.
- Enhance feature engineering for identity tracking (e.g., velocity across multiple devices).

## Docker Configuration (Added on Day 1)

The application has been containerized for reliable and isolated local development. 

### Prerequisites
- Docker
- Docker Compose

### Starting the Application
Make sure you have an `.env` file (you can copy `.env.example`).
To build and start the application, run:
```bash
docker compose up --build
```

### Accessing the Application
- **Frontend URL:** [http://localhost:3000](http://localhost:3000)
- **Backend API URL:** [http://localhost:8000](http://localhost:8000)
- **Backend Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Stopping the Application
To stop the application, run:
```bash
docker compose down
```
*(Note: A persistent local volume `backend_data` is configured in `docker-compose.yml` to preserve sqlite or csv dataset files across container restarts.)*

