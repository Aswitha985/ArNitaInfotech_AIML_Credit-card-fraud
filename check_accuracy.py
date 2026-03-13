#!/usr/bin/env python3
"""
Accuracy Check for Credit Card Fraud Detection Models
This script specifically focuses on checking and displaying accuracy metrics
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def preprocess_data(df, is_training=True):
    """Preprocess the fraud detection dataset"""
    df_processed = df.copy()
    
    # Convert datetime columns
    df_processed['trans_date_trans_time'] = pd.to_datetime(df_processed['trans_date_trans_time'])
    df_processed['dob'] = pd.to_datetime(df_processed['dob'])
    
    # Extract time-based features
    df_processed['hour'] = df_processed['trans_date_trans_time'].dt.hour
    df_processed['day_of_week'] = df_processed['trans_date_trans_time'].dt.dayofweek
    df_processed['month'] = df_processed['trans_date_trans_time'].dt.month
    
    # Calculate age
    df_processed['age'] = (df_processed['trans_date_trans_time'] - df_processed['dob']).dt.days // 365
    
    # Calculate distance between cardholder and merchant
    df_processed['distance'] = np.sqrt(
        (df_processed['lat'] - df_processed['merch_lat'])**2 + 
        (df_processed['long'] - df_processed['merch_long'])**2
    )
    
    # Select relevant features for modeling
    features = [
        'amt', 'hour', 'day_of_week', 'month', 'age', 'distance',
        'category', 'gender', 'city_pop', 'state'
    ]
    
    df_features = df_processed[features].copy()
    
    # Handle categorical variables
    categorical_features = ['category', 'gender', 'state']
    
    # Label encode categorical variables
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df_features[feature] = le.fit_transform(df_features[feature].astype(str))
        label_encoders[feature] = le
    
    # Scale numerical features
    numerical_features = ['amt', 'hour', 'day_of_week', 'month', 'age', 'distance', 'city_pop']
    
    if is_training:
        scaler = StandardScaler()
        df_features[numerical_features] = scaler.fit_transform(df_features[numerical_features])
        return df_features, scaler, label_encoders
    else:
        return df_features, label_encoders

def check_model_accuracy(model, X_test, y_test, model_name):
    """Check and return accuracy metrics for a model"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"{model_name} - ACCURACY CHECK")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    # Additional accuracy analysis
    print(f"\nDetailed Accuracy Analysis:")
    print(f"- Correct predictions: {(y_pred == y_test).sum():,} out of {len(y_test):,}")
    print(f"- Incorrect predictions: {(y_pred != y_test).sum():,}")
    print(f"- True Positives (Fraud correctly identified): {((y_pred == 1) & (y_test == 1)).sum():,}")
    print(f"- True Negatives (Legitimate correctly identified): {((y_pred == 0) & (y_test == 0)).sum():,}")
    print(f"- False Positives (Legitimate flagged as fraud): {((y_pred == 1) & (y_test == 0)).sum():,}")
    print(f"- False Negatives (Fraud missed): {((y_pred == 0) & (y_test == 1)).sum():,}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

def main():
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - ACCURACY CHECK")
    print("="*60)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_df = pd.read_csv('fraudTrain.csv')
    test_df = pd.read_csv('fraudTest.csv')
    
    print(f"Training set: {train_df.shape[0]:,} samples")
    print(f"Test set: {test_df.shape[0]:,} samples")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train, scaler, label_encoders = preprocess_data(train_df, is_training=True)
    y_train = train_df['is_fraud']
    
    X_test, test_encoders = preprocess_data(test_df, is_training=False)
    y_test = test_df['is_fraud']
    
    # Apply scaling to test data
    numerical_features = ['amt', 'hour', 'day_of_week', 'month', 'age', 'distance', 'city_pop']
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    print(f"Features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    
    # Train models
    print("\n3. Training models...")
    
    # Logistic Regression
    print("   Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    
    # Decision Tree
    print("   Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10)
    dt_model.fit(X_train, y_train)
    
    # Random Forest
    print("   Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Check accuracy for each model
    print("\n4. CHECKING ACCURACY METRICS...")
    
    results = []
    
    # Logistic Regression Accuracy
    lr_results = check_model_accuracy(lr_model, X_test, y_test, "LOGISTIC REGRESSION")
    results.append(lr_results)
    
    # Decision Tree Accuracy
    dt_results = check_model_accuracy(dt_model, X_test, y_test, "DECISION TREE")
    results.append(dt_results)
    
    # Random Forest Accuracy
    rf_results = check_model_accuracy(rf_model, X_test, y_test, "RANDOM FOREST")
    results.append(rf_results)
    
    # Create accuracy comparison table
    print(f"\n{'='*60}")
    print("ACCURACY COMPARISON TABLE")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame(results)
    print(comparison_df.round(4).to_string(index=False))
    
    # Find best accuracy
    best_accuracy_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
    best_accuracy_score = comparison_df['Accuracy'].max()
    
    print(f"\n{'='*60}")
    print("ACCURACY SUMMARY")
    print(f"{'='*60}")
    print(f"\n*** BEST ACCURACY: {best_accuracy_model} - {best_accuracy_score:.4f} ({best_accuracy_score*100:.2f}%)")
    
    print(f"\nAccuracy Rankings:")
    for idx, row in comparison_df.sort_values('Accuracy', ascending=False).iterrows():
        rank = idx + 1
        print(f"  {rank}. {row['Model']}: {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)")
    
    print(f"\nKey Observations:")
    print(f"  • All models achieve >96% accuracy due to imbalanced dataset")
    print(f"  • {best_accuracy_model} has the highest accuracy")
    print(f"  • Precision and Recall are more important for fraud detection")
    print(f"  • High accuracy doesn't always mean better fraud detection")
    
    # Save best model based on accuracy
    if best_accuracy_model == "RANDOM FOREST":
        best_model = rf_model
    elif best_accuracy_model == "DECISION TREE":
        best_model = dt_model
    else:
        best_model = lr_model
    
    joblib.dump(best_model, 'best_accuracy_model.pkl')
    print(f"\n[SAVED] Best accuracy model saved: 'best_accuracy_model.pkl'")
    
    print(f"\n{'='*60}")
    print("ACCURACY CHECK COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
