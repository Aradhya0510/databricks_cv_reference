# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Monitoring
# MAGIC 
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Set up monitoring for the deployed DETR model
# MAGIC 2. Track model performance metrics
# MAGIC 3. Monitor data drift and model drift
# MAGIC 4. Set up alerts for model degradation
# MAGIC 5. Generate monitoring reports

# COMMAND ----------

import os
import yaml
import mlflow
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Import our modules
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.monitoring.drift import DriftDetector
from src.monitoring.performance import PerformanceMonitor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration and Model

# COMMAND ----------

# Load best configuration
config_path = "/dbfs/FileStore/configs/detection_best.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load model from MLflow
model_uri = f"models:/detr_coco_detection/Production"
model = mlflow.pytorch.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Monitoring Components

# COMMAND ----------

# Initialize drift detector
drift_detector = DriftDetector(
    reference_data_path=config["data"]["data_path"],
    window_size=1000,  # Number of samples to consider for drift detection
    drift_threshold=0.1  # Threshold for drift detection
)

# Initialize performance monitor
performance_monitor = PerformanceMonitor(
    model=model,
    config=config,
    metrics_window=100  # Number of predictions to consider for performance metrics
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor Model Performance

# COMMAND ----------

# Get recent predictions from endpoint
endpoint_name = "detr-coco-endpoint"
recent_predictions = mlflow.deployments.get_endpoint(endpoint_name).predictions

# Calculate performance metrics
performance_metrics = performance_monitor.calculate_metrics(recent_predictions)

# Plot performance metrics over time
plt.figure(figsize=(15, 5))

# Plot mAP
plt.subplot(1, 3, 1)
plt.plot(performance_metrics["timestamp"], performance_metrics["map"])
plt.title("mAP Over Time")
plt.xlabel("Time")
plt.ylabel("mAP")
plt.xticks(rotation=45)

# Plot average confidence
plt.subplot(1, 3, 2)
plt.plot(performance_metrics["timestamp"], performance_metrics["avg_confidence"])
plt.title("Average Confidence Over Time")
plt.xlabel("Time")
plt.ylabel("Confidence")
plt.xticks(rotation=45)

# Plot number of detections
plt.subplot(1, 3, 3)
plt.plot(performance_metrics["timestamp"], performance_metrics["num_detections"])
plt.title("Number of Detections Over Time")
plt.xlabel("Time")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor Data Drift

# COMMAND ----------

# Get recent input data
recent_data = mlflow.deployments.get_endpoint(endpoint_name).input_data

# Calculate drift metrics
drift_metrics = drift_detector.calculate_drift(recent_data)

# Plot drift metrics
plt.figure(figsize=(15, 5))

# Plot feature drift
plt.subplot(1, 3, 1)
plt.bar(drift_metrics["features"], drift_metrics["feature_drift"])
plt.title("Feature Drift")
plt.xlabel("Feature")
plt.ylabel("Drift Score")
plt.xticks(rotation=45)

# Plot distribution drift
plt.subplot(1, 3, 2)
plt.plot(drift_metrics["timestamp"], drift_metrics["distribution_drift"])
plt.title("Distribution Drift Over Time")
plt.xlabel("Time")
plt.ylabel("Drift Score")
plt.xticks(rotation=45)

# Plot concept drift
plt.subplot(1, 3, 3)
plt.plot(drift_metrics["timestamp"], drift_metrics["concept_drift"])
plt.title("Concept Drift Over Time")
plt.xlabel("Time")
plt.ylabel("Drift Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up Alerts

# COMMAND ----------

# Define alert thresholds
alert_thresholds = {
    "performance": {
        "map_threshold": 0.5,
        "confidence_threshold": 0.7
    },
    "drift": {
        "feature_drift_threshold": 0.2,
        "distribution_drift_threshold": 0.15,
        "concept_drift_threshold": 0.1
    }
}

# Check for alerts
alerts = []

# Check performance alerts
if performance_metrics["map"].iloc[-1] < alert_thresholds["performance"]["map_threshold"]:
    alerts.append({
        "type": "performance",
        "metric": "mAP",
        "value": performance_metrics["map"].iloc[-1],
        "threshold": alert_thresholds["performance"]["map_threshold"]
    })

if performance_metrics["avg_confidence"].iloc[-1] < alert_thresholds["performance"]["confidence_threshold"]:
    alerts.append({
        "type": "performance",
        "metric": "confidence",
        "value": performance_metrics["avg_confidence"].iloc[-1],
        "threshold": alert_thresholds["performance"]["confidence_threshold"]
    })

# Check drift alerts
if drift_metrics["feature_drift"].max() > alert_thresholds["drift"]["feature_drift_threshold"]:
    alerts.append({
        "type": "drift",
        "metric": "feature_drift",
        "value": drift_metrics["feature_drift"].max(),
        "threshold": alert_thresholds["drift"]["feature_drift_threshold"]
    })

if drift_metrics["distribution_drift"].iloc[-1] > alert_thresholds["drift"]["distribution_drift_threshold"]:
    alerts.append({
        "type": "drift",
        "metric": "distribution_drift",
        "value": drift_metrics["distribution_drift"].iloc[-1],
        "threshold": alert_thresholds["drift"]["distribution_drift_threshold"]
    })

if drift_metrics["concept_drift"].iloc[-1] > alert_thresholds["drift"]["concept_drift_threshold"]:
    alerts.append({
        "type": "drift",
        "metric": "concept_drift",
        "value": drift_metrics["concept_drift"].iloc[-1],
        "threshold": alert_thresholds["drift"]["concept_drift_threshold"]
    })

# Display alerts
if alerts:
    print("Active Alerts:")
    for alert in alerts:
        print(f"- {alert['type'].title()} Alert: {alert['metric']} = {alert['value']:.4f} (threshold: {alert['threshold']:.4f})")
else:
    print("No active alerts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Monitoring Report

# COMMAND ----------

# Create monitoring report
report = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": "detr_coco_detection",
    "endpoint_name": endpoint_name,
    "performance_metrics": {
        "current_map": performance_metrics["map"].iloc[-1],
        "current_confidence": performance_metrics["avg_confidence"].iloc[-1],
        "current_detections": performance_metrics["num_detections"].iloc[-1]
    },
    "drift_metrics": {
        "feature_drift": drift_metrics["feature_drift"].max(),
        "distribution_drift": drift_metrics["distribution_drift"].iloc[-1],
        "concept_drift": drift_metrics["concept_drift"].iloc[-1]
    },
    "alerts": alerts
}

# Save report
report_path = "/dbfs/FileStore/monitoring/report.json"
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

# Log report to MLflow
with mlflow.start_run(run_name="model_monitoring") as run:
    mlflow.log_artifact(report_path)
    mlflow.log_metrics({
        "current_map": report["performance_metrics"]["current_map"],
        "current_confidence": report["performance_metrics"]["current_confidence"],
        "feature_drift": report["drift_metrics"]["feature_drift"],
        "distribution_drift": report["drift_metrics"]["distribution_drift"],
        "concept_drift": report["drift_metrics"]["concept_drift"]
    })

print(f"Monitoring report generated and saved to {report_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Model monitoring setup completed:
# MAGIC - Performance metrics tracked
# MAGIC - Data drift monitored
# MAGIC - Alerts configured
# MAGIC - Monitoring report generated
# MAGIC 
# MAGIC The model is now ready for production use with continuous monitoring. 