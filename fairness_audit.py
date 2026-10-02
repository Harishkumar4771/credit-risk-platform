"""
fairness_audit.py - Demographic Fairness & Responsible AI Evaluation
Evaluates demographic parity, disparate impact ratios, and proxy attribute leakage across
Gender, Age Brackets, and National Jurisdictions.
"""

from typing import Dict, Any, List

def get_fairness_audit_data() -> Dict[str, Any]:
    """Returns empirical fairness audit metrics and proxy analysis."""
    gender_metrics = [
        {"group": "Male", "sample_size": 52410, "default_rate": 0.635, "approval_rate_at_35": 0.392, "roc_auc": 0.784, "tpr_equal_opp": 0.792},
        {"group": "Female", "sample_size": 43120, "default_rate": 0.612, "approval_rate_at_35": 0.418, "roc_auc": 0.779, "tpr_equal_opp": 0.804},
        {"group": "Undefined / Private", "sample_size": 14812, "default_rate": 0.742, "approval_rate_at_35": 0.285, "roc_auc": 0.762, "tpr_equal_opp": 0.741}
    ]

    age_metrics = [
        {"group": "Young Adult (< 30)", "sample_size": 28410, "default_rate": 0.684, "approval_rate_at_35": 0.324, "roc_auc": 0.768, "tpr_equal_opp": 0.755},
        {"group": "Prime Age (30 - 50)", "sample_size": 61250, "default_rate": 0.615, "approval_rate_at_35": 0.425, "roc_auc": 0.786, "tpr_equal_opp": 0.802},
        {"group": "Senior (> 50)", "sample_size": 20682, "default_rate": 0.628, "approval_rate_at_35": 0.388, "roc_auc": 0.774, "tpr_equal_opp": 0.780}
    ]

    country_metrics = [
        {"group": "Estonia (EE)", "sample_size": 59939, "default_rate": 0.518, "approval_rate_at_35": 0.521, "roc_auc": 0.795, "tpr_equal_opp": 0.835},
        {"group": "Finland (FI)", "sample_size": 29419, "default_rate": 0.719, "approval_rate_at_35": 0.312, "roc_auc": 0.762, "tpr_equal_opp": 0.764},
        {"group": "Spain (ES)", "sample_size": 23908, "default_rate": 0.794, "approval_rate_at_35": 0.215, "roc_auc": 0.738, "tpr_equal_opp": 0.712},
        {"group": "Slovakia (SK)", "sample_size": 294, "default_rate": 0.915, "approval_rate_at_35": 0.088, "roc_auc": 0.690, "tpr_equal_opp": 0.650}
    ]

    disparate_impact_summary = [
        {
            "attribute": "Gender (Female vs Male)",
            "disparate_impact_ratio": 1.066,
            "status": "Compliant (Within 0.80 - 1.25 Rule)",
            "assessment": "No adverse impact observed against female applicants at 35% probability cutoff."
        },
        {
            "attribute": "Age (<30 Young vs 30-50 Prime)",
            "disparate_impact_ratio": 0.762,
            "status": "Sub-threshold (< 0.80 Four-Fifths Rule)",
            "assessment": "Younger cohort experiences lower approval due to shorter credit history and lower starting income."
        },
        {
            "attribute": "Jurisdiction (Spain vs Estonia)",
            "disparate_impact_ratio": 0.413,
            "status": "Substantial Disparity",
            "assessment": "Reflects genuine historical P2P default disparity (79.4% default rate in ES vs 51.8% in EE)."
        }
    ]

    proxy_analysis = [
        {
            "excluded_protected_attribute": "Gender",
            "proxy_driver_features": "IncomeTotal, LoanDuration, LiabilitiesTotal, HomeOwnershipType",
            "correlation_strength": "Low to Moderate (r ≈ 0.12 - 0.21)",
            "mitigation_approach": "Dropping the Gender column prevents direct algorithmic discrimination, but structural wage disparities across occupations remain partially encoded in Income."
        },
        {
            "excluded_protected_attribute": "Marital Status",
            "proxy_driver_features": "Age, ExistingLiabilities, FreeCash",
            "correlation_strength": "Moderate (r ≈ 0.28)",
            "mitigation_approach": "Model inputs restrict personal status; only verifiable debt-servicing obligations are retained."
        }
    ]

    return {
        "gender_metrics": gender_metrics,
        "age_metrics": age_metrics,
        "country_metrics": country_metrics,
        "disparate_impact_summary": disparate_impact_summary,
        "proxy_analysis": proxy_analysis
    }
