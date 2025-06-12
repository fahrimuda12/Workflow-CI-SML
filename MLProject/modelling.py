import pandas as pd
import mlflow
import numpy as np 
import mlflow.sklearn
import dagshub
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # tidak bisa memakai dagshub jika dilakukan workflow CI/
    # dagshub.init(repo_owner='fahrimuda12', repo_name='heart-disease', mlflow=True)

    # Inisialisasi MLflow autolog
    mlflow.sklearn.autolog()

    # Set MLflow Tracking URI
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Nama eksperimen yang ingin dicari
    experiment_name = "model_heart_disease_failure"

    # Create a new MLflow Experiment
    # Ensure experiment name is valid (no spaces or special characters)
    experiment_name = experiment_name.replace(" ", "_").replace("-", "_")
    mlflow.set_experiment(experiment_name)

    # Optional: Lihat apakah experiment berhasil di-set
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print("Experiment ID:", experiment.experiment_id)

    # Memuat dataset 
    data = pd.read_csv('./dataset/failure_heart_preprocessing.csv', sep=',')

    print(data.info())

    # Preprocessing (ubah ini sesuai dataset)
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    # Konversi eksplisit ke integer
    y = y.astype(int)

    print("Unique labels:", y.unique())
    print("Label dtype:", y.dtype)

    # Dummy encoding if needed
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === 3. Online Training Setup ===
    batch_size = 60
    n_batches = int(np.ceil(len(X_train) / batch_size))
    clf = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, max_iter=10000, random_state=42)


    # -----------------------------
    # STEP 3: MLflow Training
    # -----------------------------
    # 4. MLflow Logging
    with mlflow.start_run():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(X_train))
            X_batch, y_batch = X_train[start:end], y_train[start:end]
            
            if i == 0:
                clf.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
            else:
                clf.partial_fit(X_batch, y_batch)

        # Inference setelah semua batch
        y_pred = clf.predict(X_test)

        # Jika ingin, bisa tambahkan evaluasi manual
        score = clf.score(X_test, y_test)
        print(f"Test accuracy: {score}")


        # Contoh input
        input_example = X_train.iloc[:2].copy()
        # Pastikan tidak ada NaN pada input_example
        input_example = input_example.fillna(0)

        # Signature model
        signature = infer_signature(X_train, clf.predict(X_train))

        # Log final model
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model_heart_disease_failure",
            input_example=input_example,
            signature=signature
        )

        mlflow.sklearn.save_model(clf, path="model")
