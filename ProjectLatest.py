import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------- Paths ---------
csv_path = "projectdatasetxx.csv"
model_path = "risk_rf_model.pkl"
scaler_path = "feature_scaler.pkl"

# --------- Train Model with Evaluation Metrics ---------
def train_model():
    df = pd.read_csv(csv_path)

    if 'Risk_Score' not in df.columns:
        # Increase weight for dependency and reduce for independence
        df['Risk_Score'] = (df['Task_dependency_values'] * 0.8 - df['Independent_values'] * 0.5) * 10
        df['Risk_Score'] = df['Risk_Score'].clip(lower=0)  # ensure non-negative scores


    features = df[['Total_tasks', 'Task_dependency_values', 'Independent_values']]
    labels = df['Risk_Score']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # --------- Evaluation Metrics ---------
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler

# --------- Load Model (force retrain every time to get metrics) ---------
def load_model():
    return train_model()  # <- Always retrain to print metrics

# --------- Predict and Generate DAG ---------
def predict_and_generate_dag(project_name, task_names, dep_value, indep_value):
    tasks = [task.strip() for task in task_names.split(',')]
    total_tasks = len(tasks)

    model, scaler = load_model()

    input_df = pd.DataFrame([[total_tasks, dep_value, indep_value]], 
                            columns=['Total_tasks', 'Task_dependency_values', 'Independent_values'])
    input_scaled = scaler.transform(input_df)

    risk_score = model.predict(input_scaled)[0]

    if risk_score >= 5:
        risk_level = "High Risk"
    elif risk_score >= 4:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # --------- DAG Generation ---------
    G = nx.DiGraph()
    for task in tasks:
        G.add_node(task)
    for i in range(len(tasks) - 1):
        G.add_edge(tasks[i], tasks[i + 1], weight=round(dep_value / max(1, total_tasks - 1), 2))

    pos = nx.spring_layout(G, seed=42)
    edge_weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue',
            font_size=10, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=9)

    plt.title(f"{project_name} - Risk Score: {risk_score:.2f} → {risk_level}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("dag_output.png")
    plt.show()

    return round(risk_score, 2), risk_level

# --------- Example Run ---------
if __name__ == "__main__":
    predict_and_generate_dag(
        project_name="Website Redesign",
        task_names="Design,UI research,Prototyping,Coding,Testing",
        dep_value=0.75,
        indep_value=0.25
    )
