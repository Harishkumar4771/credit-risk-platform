# AI Credit Risk & Financial Inclusion Platform

An end-to-end Machine Learning and Explainable AI (XAI) underwriting web platform built on the **Bondora Peer-to-Peer Lending dataset** with **Neuro-Symbolic Expert Policies**, **SHAP**, and **LIME**.

---

## 🌟 Key Features

1. **Zero Data Leakage Underwriting**:
   - Strictly audits 112 raw Bondora columns and retains only application-time predictors.
   - Eliminates post-origination repayment/recovery variables and platform-generated rating leakage.
2. **Interactive Credit Risk Assessment**:
   - Computes Probability of Default (PD), credit risk band (A through F), and underwriting decision (Approved / Conditional Review / Declined).
   - Flags **Financial Inclusion** candidates (responsible thin-file applicants with healthy cash flow).
3. **Triple-Layer Explainable AI (XAI)**:
   - **SHAP (SHapley Additive exPlanations)**: Tree attribution with strict mathematical additivity check ($\sum \phi_i + \phi_0 = f(x)$).
   - **LIME (Local Interpretable Model-agnostic Explanations)**: Neighborhood perturbations in the *unencoded original feature space* to prevent categorical collisions.
   - **Neuro-Symbolic Reasoning Engine**: First-Order Logic expert rulebook providing human-readable syllogistic audits and checking consensus against empirical gradient boosting.
4. **Fairness & Responsible AI Audit**:
   - Evaluates Demographic Parity, Disparate Impact (Four-Fifths Rule), and Equal Opportunity across Gender, Age Groups, and Country (Estonia, Finland, Spain, Slovakia).
   - Analyzes latent proxy variable leakage.
5. **Model Card & Benchmark Leaderboard**:
   - 7 candidate algorithms benchmarked with 5-fold cross-validation.
   - Transparently reports the temporal drift drop (0.781 random test ROC-AUC vs 0.715 out-of-time future validation).

---

## 🚀 How to Run the Website

### Option 1: Run with Streamlit (Recommended)
Open your terminal inside the project directory:

```bash
cd "C:\Users\Akshita\.gemini\antigravity\scratch\credit_risk_platform"
streamlit run app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.

### Option 2: Uploading Google Colab Artifacts
Whenever you download the trained artifacts from your Google Colab notebook:
- `default_risk_pipeline.joblib`
- `symbolic_scorecard.joblib`

Simply place them in this folder or upload them directly via the **⚙️ Artifact Settings** tab in the web interface.

---

## 📁 Project Structure

```text
credit_risk_platform/
├── app.py                     # Main interactive Streamlit Web Application
├── loan_utils.py              # Preprocessing, 0-leakage feature engineering & model pipeline
├── neuro_symbolic.py          # First-Order Logic expert rule base & syllogisms
├── xai_engine.py              # SHAP TreeAttribution & LIME local perturbation surrogates
├── fairness_audit.py          # Subgroup parity, disparate impact & proxy variable audit
├── model_card.json            # Model specifications, validation metrics & benchmarks
├── sample_applicants.csv      # Pre-configured test applicant archetypes
├── requirements.txt           # Python library dependencies
└── README.md                  # Documentation and execution instructions
```
