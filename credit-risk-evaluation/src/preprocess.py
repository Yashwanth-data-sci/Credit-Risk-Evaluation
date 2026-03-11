"""
Credit Risk Evaluation — Preprocessing Utilities
Author: Bandaru Yashwanth | B.Sc. Actuarial Science, Amity University Noida
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_and_impute(filepath):
    """Load CSV and handle missing values with median imputation."""
    df = pd.read_csv(filepath)
    cols = ['credit_score', 'employment_years', 'num_credit_lines']
    imputer = SimpleImputer(strategy='median')
    df[cols] = imputer.fit_transform(df[cols])
    return df


def engineer_features(df):
    """
    Create actuarial-style risk features.
    Mirrors real-world underwriting ratios used by banks & insurers.
    """
    df = df.copy()

    # Risk ratios
    df['debt_to_income_ratio']  = (df['existing_debt']  / df['annual_income']).round(4)
    df['loan_to_income_ratio']  = (df['loan_amount']    / df['annual_income']).round(4)
    df['monthly_emi_burden']    = (df['loan_amount']    / df['loan_term_months']).round(2)
    df['emi_to_monthly_income'] = (df['monthly_emi_burden'] / (df['annual_income'] / 12)).round(4)
    df['financial_stress']      = (
        0.4 * df['debt_to_income_ratio'] +
        0.3 * df['loan_to_income_ratio'] +
        0.3 * df['emi_to_monthly_income']
    ).round(4)

    # Actuarial tier classification
    df['credit_tier'] = pd.cut(df['credit_score'],
                                bins=[0, 550, 620, 680, 750, 900],
                                labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
    df['income_tier'] = pd.cut(df['annual_income'],
                                bins=[0, 25000, 50000, 100000, 200001],
                                labels=['Low', 'Lower-Mid', 'Upper-Mid', 'High'])
    df['age_group']   = pd.cut(df['age'],
                                bins=[0, 25, 35, 45, 55, 100],
                                labels=['Young Adult', 'Early Career', 'Mid Career', 'Senior', 'Pre-Retirement'])

    # Ordinal encode tiers
    ordinal_maps = {
        'credit_tier': {'Very Poor': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Excellent': 5},
        'income_tier': {'Low': 1, 'Lower-Mid': 2, 'Upper-Mid': 3, 'High': 4},
        'age_group':   {'Young Adult': 1, 'Early Career': 2, 'Mid Career': 3,
                        'Senior': 4, 'Pre-Retirement': 5}
    }
    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping)

    return df


def encode_and_split(df, target='default', test_size=0.2, random_state=42):
    """One-hot encode categoricals and split into train/test."""
    from sklearn.model_selection import train_test_split

    df_enc = pd.get_dummies(
        df.drop(columns=['customer_id']),
        columns=['employment_type', 'education', 'property_ownership', 'loan_purpose'],
        drop_first=True
    )
    X = df_enc.drop(columns=[target])
    y = df_enc[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def assign_risk_tier(probability):
    """
    Convert default probability to actuarial risk tier.
    Returns: (tier, recommended_action)
    """
    if probability < 0.10:
        return 'LOW RISK', 'Auto-Approve'
    elif probability < 0.25:
        return 'MODERATE RISK', 'Standard Review'
    elif probability < 0.50:
        return 'HIGH RISK', 'Senior Review Required'
    else:
        return 'CRITICAL RISK', 'Decline / Require Collateral'
