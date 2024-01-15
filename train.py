import joblib
import os
import xgboost as xgb
from sklearn.metrics import f1_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import hopsworks
from config import CONFIG

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


def training_model(X_train, X_test, y_train, y_test):
    xgboost_model = xgb.XGBClassifier()
    xgboost_model.fit(X_train, y_train)
    score = f1_score(
        y_test,
        xgboost_model.predict(X_test),
        average="macro",
    )
    print(f"⛳️ F1 score: {score}")
    return xgboost_model, score


def save_best_model(project, xgboost_model, score, input_schema, output_schema):
    if os.path.isdir(CONFIG.MODEL_DIR) == False:
        os.mkdir(CONFIG.MODEL_DIR)

    mr = project.get_model_registry()
    # Creating a model schema
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    model = mr.python.create_model(
        name=CONFIG.MODEL_NAME,
        metrics={"f1_score": score},
        description="XGB for Credit Scores Project",
        input_example=X_train.sample(),
        model_schema=model_schema,
    )

    # Saving the trained XGBoost model as a joblib file in the model directory
    joblib.dump(xgboost_model, CONFIG.MODEL_DIR + "/credit_scores_model.pkl")
    conf_matrix = confusion_matrix(
        y_test,
        xgboost_model.predict(X_test),
    )
    figure_cm = plt.figure(figsize=(10, 7))
    figure_cm = sns.heatmap(
        conf_matrix,
        annot=True,
        annot_kws={"size": 14},
        fmt=".10g",
    )
    figure_imp = plot_importance(
        xgboost_model,
        max_num_features=10,
        importance_type="weight",
    )
    figure_cm.figure.savefig(CONFIG.MODEL_DIR + "/confusion_matrix.png")
    figure_imp.figure.savefig(CONFIG.MODEL_DIR + "/feature_importance.png")
    model.save(CONFIG.MODEL_DIR)


if __name__ == "__main__":
    project = hopsworks.login()

    fs = project.get_feature_store()
    applications_fg = fs.get_feature_group(
        name=CONFIG.FEATURE_GROUP,
        version=1,
    )

    # Retrieving the Label Encoder transformation function from Featuretools

    # Creating a dictionary of transformation functions, where each categorical column is associated with the Label Encoder
    query = applications_fg.select_all()
    feature_view = fs.get_or_create_feature_view(
        name=CONFIG.FEATURE_VIEW,
        version=1,
        labels=["Rich_class"],
        query=query,
    )
    X_train, X_test, y_train, y_test = feature_view.train_test_split(
        test_size=0.2,
    )
    xgboost_model, score = training_model(X_train, X_test, y_train, y_test)
    input_schema = Schema(X_train.values)
    output_schema = Schema(y_train)
    save_best_model(project, xgboost_model, score, input_schema, output_schema)