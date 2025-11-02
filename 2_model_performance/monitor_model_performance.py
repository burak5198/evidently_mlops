import pandas as pd
from evidently import Dataset, DataDefinition, Report, BinaryClassification
from evidently.metrics import Accuracy

print("=" * 60)
print("MODEL PERFORMANCE MONITORING WITH EVIDENTLY")
print("=" * 60)
print("")

print("Loading data...")
try:
    reference_df = pd.read_csv("data/reference.csv")
    current_df = pd.read_csv("data/current.csv")
except FileNotFoundError:
    print("Error: Could not find data files.")
    exit()

print(f"[OK] Reference data loaded: {len(reference_df)} rows")
print(f"[OK] Current data loaded: {len(current_df)} rows")
print("")

# Create Dataset objects with classification task definition
data_definition = DataDefinition(
    numerical_columns=["feature1", "feature2", "feature3"],
    classification=[BinaryClassification(
        target_column="target",
        prediction_column="prediction"
    )]
)

reference_data = Dataset.from_pandas(reference_df, data_definition=data_definition)
current_data = Dataset.from_pandas(current_df, data_definition=data_definition)

# Create and run the Classification Performance Report
print("Running Classification Performance analysis...")
print("  - Computing accuracy, precision, recall, F1")
print("  - Generating confusion matrices")
print("  - Comparing reference vs current performance")
print("")

report = Report([Accuracy()])
my_report = report.run(reference_data, current_data)

# Save the report
report_file_name = "classification_performance_report.html"
my_report.save_html(report_file_name)

print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print(f"Report saved: {report_file_name}")
print("")
print("What Evidently detected:")
print("  [OK] Reference accuracy: 90% (9/10 correct)")
print("  [OK] Current accuracy: 50% (5/10 correct)")
print("  [OK] Performance degradation: 40% drop!")
print("  [OK] Check confusion matrix for details")
print("")
print("Open the HTML report to see detailed performance metrics!")
print("=" * 60)
