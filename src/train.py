import numpy as np, pandas as pd, joblib, os
import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1) สร้างข้อมูลจำลอง
rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "area": rng.uniform(30, 300, n),
    "bedrooms": rng.integers(1, 6, n),
    "location": rng.choice(["city", "suburb", "rural"], n),
})
loc_price = df["location"].map({"city": 90000, "suburb": 50000, "rural": 25000})
df["price"] = df["area"] * loc_price + df["bedrooms"] * 300000 + rng.normal(0, 500000, n)
X, y = df[["area", "bedrooms", "location"]], df["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pre = ColumnTransformer([("loc", OneHotEncoder(handle_unknown="ignore"), ["location"])], remainder="passthrough")
models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
}
mlflow.set_experiment("house-price")
results = {}

# 2) Train ทุกโมเดล + บันทึกลง MLflow
for name, algo in models.items():
    with mlflow.start_run(run_name=name) as run:
        pipe = Pipeline([("pre", pre), ("model", algo)]).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rmse = mean_squared_error(y_te, pred) ** 0.5
        mlflow.log_param("algorithm", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2_score(y_te, pred))
        
        # 🟢 แก้ไขบรรทัดนี้: เพิ่ม serialization_format="cloudpickle" เพื่อแก้ Error
        mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")
        
        results[name] = (rmse, run.info.run_id, pipe)
        print(name, "RMSE =", round(rmse, 2))

# 3) เลือกตัวที่ RMSE ต่ำสุด แล้วลงทะเบียน
best = min(results, key=lambda k: results[k][0])
rmse, run_id, best_pipe = results[best]
mv = mlflow.register_model(f"runs:/{run_id}/model", "house-price-model")
MlflowClient().set_registered_model_alias("house-price-model", "production", mv.version)
print(f"Registered house-price-model:v{mv.version} ({best})")

# 4) export ไว้ให้ Docker ใช้
os.makedirs("model", exist_ok=True)
joblib.dump(best_pipe, "model/model.joblib")