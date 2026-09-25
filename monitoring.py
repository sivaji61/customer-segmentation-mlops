import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Reference data
reference_data = pd.read_csv("data/customers.csv")

# Current data
current_data = pd.read_csv("data/customers.csv")

# Create monitoring report
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

# Run report
result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save HTML report
result.save_html("data_drift_report.html")

print("Data drift monitoring report generated successfully!")