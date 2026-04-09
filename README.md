# 💼 Data Science Salary Predictor

A machine learning web application that predicts data science salaries based on
role, experience level, location, and company attributes.

Built and deployed with Streamlit. Trained on 607 real-world data science job
records spanning 2020–2022 across 50+ job titles and 40+ countries.

---

## 🚀 Live App

> Run locally with: `streamlit run app/app.py`

---

## 📊 Project Overview

| | |
|---|---|
| **Dataset** | Data Science Job Salaries (Kaggle) |
| **Records** | 607 rows × 9 features |
| **Target** | Salary in USD |
| **Model** | Random Forest Regressor |
| **R² Score** | 0.548 |
| **MAE** | $29,360 |
| **Tools** | Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit |

---

## 🔍 Key Findings

- **US residency is the #1 salary driver** (feature importance: 0.407) — a
  US-based entry-level role pays more than a non-US executive role on average
- **Experience level** is the second strongest predictor (0.151)
- **Median salaries grew 59%** from $75K (2020) to $120K (2022)
- **Hybrid roles pay the least** ($80K avg) — fully remote roles pay the most ($122K avg)
- **Principal Data Scientist** tops the salary chart at $215K average

---

## 📁 Project Structure
job-salary-predictor/
├── app/
│   └── app.py                  # Streamlit web application
├── data/
│   ├── raw/ds_salaries.csv     # Original dataset
│   └── processed/cleaned_salaries.csv
├── models/
│   ├── salary_model.pkl        # Trained Random Forest model
│   └── feature_columns.pkl    # Feature column names for inference
├── notebooks/
│   └── salary_predictor.ipynb # Full analysis notebook
├── visuals/                    # 10 EDA & model visualizations
├── requirements.txt
└── README.md

---

## 📈 Visualizations

| # | Chart | Insight |
|---|-------|---------|
| 1 | Salary Distribution | Right-skewed; log transform applied for ML |
| 2 | Salary by Experience | Entry $61K → Executive $199K avg |
| 3 | Top 15 Job Titles | Principal DS leads at $215K |
| 4 | Company Size & Remote | Hybrid pays least; remote pays most |
| 5 | Salary Trends 2020–2022 | 59% median increase in 2 years |
| 6 | Geographic Analysis | US companies dominate volume and pay |
| 7 | Correlation Heatmap | US company (0.53) and experience (0.48) top features |
| 8 | Experience × Location | US entry-level earns more than non-US executive |
| 9 | Feature Importance | is_us_resident drives 40.7% of model decisions |
| 10 | Actual vs Predicted | 75% of predictions within $40K of actual |

---

## 🤖 ML Workflow

1. **Exploratory Data Analysis** — 8 professional visualizations
2. **Feature Engineering** — log-transform target, ordinal encode experience,
   flag US residence/company, group rare job titles
3. **Model Selection** — compared Random Forest vs Gradient Boosting
4. **Improvement** — adding `is_us_resident` improved R² from 0.454 → 0.548
5. **Evaluation** — residuals analysis, actual vs predicted, error breakdown

---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/rohithphani/job-salary-predictor.git
cd job-salary-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app/app.py
```

---

## ⚠️ Model Limitations

- Dataset covers 2020–2022 only — salary trends may differ today
- 58.5% of records are US-based — model is most accurate for US roles
- Executive tier has only 26 records — predictions less reliable at that level
- Salary reflects reported figures; actual compensation may include equity/bonuses

---

## 👤 Author

**Rohith Phani Gavirneni**
M.S. Engineering Data Science — University of Houston
GitHub: [rohithphani](https://github.com/rohithphani)