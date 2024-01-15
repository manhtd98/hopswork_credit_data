import joblib
import hopsworks
from config import CONFIG


def load_model(mr):
    retrieved_model = mr.get_model(
        name=CONFIG.MODEL_NAME,
        version=1,
    )

    saved_model_dir = retrieved_model.download()
    retrieved_xgboost_model = joblib.load(saved_model_dir + "/credit_scores_model.pkl")
    return retrieved_xgboost_model


if __name__ == "__main__":
    project = hopsworks.login()
    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(
        name=CONFIG.FEATURE_VIEW,
        version=1,
    )
    mr = project.get_model_registry()
    retrieved_xgboost_model = load_model(mr)
    feature_view.init_batch_scoring(1)
    batch_data = feature_view.get_batch_data()
    predictions = retrieved_xgboost_model.predict(batch_data)
