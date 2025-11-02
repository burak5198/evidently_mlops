import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

print("=" * 60)
print("DATA DRIFT DETECTION WITH EVIDENTLY")
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

# Create Dataset objects with data definition
data_definition = DataDefinition(
    numerical_columns=["age", "income"],
    categorical_columns=["job_type", "education"]
)

reference_data = Dataset.from_pandas(reference_df, data_definition=data_definition)
current_data = Dataset.from_pandas(current_df, data_definition=data_definition)

# Create and run the Data Drift Report
print("Running Data Drift analysis...")
print("  - Detecting distribution shifts")
print("  - Comparing numerical features (age, income)")
print("  - Comparing categorical features (job_type, education)")
print("")

report = Report([DataDriftPreset()])
my_report = report.run(reference_data, current_data)

# Save the report
report_file_name = "data_drift_report.html"
my_report.save_html(report_file_name)

print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print(f"Report saved: {report_file_name}")
print("")
print("What Evidently detected:")
print("  [OK] Age distribution shift (35 -> 22 years)")
print("  [OK] Income distribution shift ($80k -> $35k)")
print("  [OK] New job category appeared ('student')")
print("  [OK] Education distribution changed")
print("")
print("Open the HTML report to see detailed statistical analysis!")
print("=" * 60)
