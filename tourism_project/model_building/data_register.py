
import pandas as pd
import mlflow
import os

def register_dataset(data_path="tourism_project/data", filename="customer_data.csv"):
    """Simulates dataset loading and registration with MLflow."""
    print(f"Loading and registering dataset: {filename}")

    # Create a dummy dataset for demonstration if not exists
    if not os.path.exists(os.path.join(data_path, filename)):
        print("Creating dummy customer data...")
        dummy_data = {
            'CustomerID': range(1, 101),
            'Age': [25 + i % 40 for i in range(100)],
            'MonthlyIncome': [2000 + i * 100 for i in range(100)],
            'ProdTaken': [i % 2 for i in range(100)]
        }
        df = pd.DataFrame(dummy_data)
        os.makedirs(data_path, exist_ok=True)
        df.to_csv(os.path.join(data_path, filename), index=False)
        print(f"Dummy data saved to {os.path.join(data_path, filename)}")
    else:
        df = pd.read_csv(os.path.join(data_path, filename))
        print(f"Dataset loaded from {os.path.join(data_path, filename)}")

    # Log the dataset as an artifact in MLflow
    with mlflow.start_run(run_name="Data_Registration"):
        mlflow.log_artifact(os.path.join(data_path, filename))
        mlflow.log_param("dataset_version", pd.Timestamp.now().strftime("%Y%m%d%H%M%S"))
        print("Dataset registered with MLflow.")

if __name__ == "__main__":
    register_dataset()
