

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import mlflow
import mlflow.sklearn
from datetime import datetime
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import dagshub
import os


def load_preprocessed_data(train_path='iris_train_preprocessed.csv', 
                           test_path='iris_test_preprocessed.csv'):
    """
    Load preprocessed data
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"Loading data from {train_path} and {test_path}...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['species'])
    y_train = train_df['species']
    X_test = test_df.drop(columns=['species'])
    y_test = test_df['species']
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, max_depth=10, 
                min_samples_split=2, min_samples_leaf=1, random_state=42):
    """
    Train Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples per leaf
        random_state: Random state
        
    Returns:
        Trained model
    """
    print("\nTraining Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        dict: Dictionary of metrics
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, classes, filename='confusion_matrix.png'):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def plot_feature_importance(model, feature_names, filename='feature_importance.png'):
    """
    Plot feature importance
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Feature importance saved to {filename}")


def main():
    """
    Main training pipeline for MLflow Project
    """
    parser = argparse.ArgumentParser(description='Train Iris Classification Model')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=10, help='Max depth of trees')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Min samples to split')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Min samples per leaf')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MLflow Project - CI/CD Model Training Pipeline")
    print("="*70)
    
    
    # For DagsHub remote tracking (uncomment when needed):
    # dagshub.init(repo_owner='josephgreffenkomala', repo_name='Membangun-model_SML', mlflow=True)
    # mlflow.set_tracking_uri("https://dagshub.com/josephgreffenkomala/Workflow-CI-dicoding.mlflow")
    
    
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Start MLflow run with nested=False to ensure fresh run
        
    # Log parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    
    # Train model
    model = train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Log metrics
    mlflow.log_metric("train_accuracy", metrics['train_accuracy'])
    mlflow.log_metric("test_accuracy", metrics['test_accuracy'])
    mlflow.log_metric("test_precision", metrics['test_precision'])
    mlflow.log_metric("test_recall", metrics['test_recall'])
    mlflow.log_metric("test_f1", metrics['test_f1'])
    
    # Create and log visualizations
    classes = sorted(y_test.unique())
    plot_confusion_matrix(metrics['confusion_matrix'], classes)
    mlflow.log_artifact('confusion_matrix.png')

    feature_names = X_train.columns.tolist()
    plot_feature_importance(model, feature_names)
    mlflow.log_artifact('feature_importance.png')
    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_train.iloc[:5],
        signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
    )
    
    # Get run info
    run = mlflow.active_run()
    print(f"\nMLflow Run ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")
    
    # Save run ID to file for CI/CD
    with open('run_id.txt', 'w') as f:
        f.write(run.info.run_id)
        
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
# docker run -p 5001:8080 josephgreffenkomala/iris-classification-mlflow:latest  