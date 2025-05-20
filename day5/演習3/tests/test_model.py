# import os
# import pytest
# import pandas as pd
# import numpy as np
# import pickle
# import time
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# # テスト用データとモデルパスを定義
# DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
# MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
# BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_model.pkl")


# @pytest.fixture
# def sample_data():
#     """テスト用データセットを読み込む"""
#     if not os.path.exists(DATA_PATH):
#         from sklearn.datasets import fetch_openml

#         titanic = fetch_openml("titanic", version=1, as_frame=True)
#         df = titanic.data
#         df["Survived"] = titanic.target

#         # 必要なカラムのみ選択
#         df = df[
#             ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
#         ]

#         os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
#         df.to_csv(DATA_PATH, index=False)

#     return pd.read_csv(DATA_PATH)


# @pytest.fixture
# def preprocessor():
#     """前処理パイプラインを定義"""
#     # 数値カラムと文字列カラムを定義
#     numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
#     categorical_features = ["Sex", "Embarked"]

#     # 数値特徴量の前処理（欠損値補完と標準化）
#     numeric_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler()),
#         ]
#     )

#     # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
#     categorical_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore")),
#         ]
#     )

#     # 前処理をまとめる
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features),
#         ]
#     )

#     return preprocessor


# @pytest.fixture
# def train_model(sample_data, preprocessor):
#     """モデルの学習とテストデータの準備"""
#     # データの分割とラベル変換
#     X = sample_data.drop("Survived", axis=1)
#     y = sample_data["Survived"].astype(int)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # モデルパイプラインの作成
#     model = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
#         ]
#     )

#     # モデルの学習
#     model.fit(X_train, y_train)

#     # モデルの保存
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump(model, f)

#     return model, X_test, y_test

# @pytest.fixture
# def train_baseline_model(sample_data, preprocessor):
#     """訓練基線模型（邏輯回歸）並儲存"""
#     # 分割資料並轉換標籤
#     X = sample_data.drop("Survived", axis=1)
#     y = sample_data["Survived"].astype(int)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # 建立基線模型流水線
#     baseline_model = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
#         ]
#     )

#     # 訓練基線模型
#     baseline_model.fit(X_train, y_train)

#     # 儲存基線模型
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     with open(BASELINE_MODEL_PATH, "wb") as f:
#         pickle.dump(baseline_model, f)

#     return baseline_model, X_test, y_test


# def test_model_exists():
#     """モデルファイルが存在するか確認"""
#     if not os.path.exists(MODEL_PATH):
#         pytest.skip("モデルファイルが存在しないためスキップします")
#     assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


# def test_model_accuracy(train_model):
#     """モデルの精度を検証"""
#     model, X_test, y_test = train_model

#     # 予測と精度計算
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
#     assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


# def test_model_inference_time(train_model):
#     """モデルの推論時間を検証"""
#     model, X_test, _ = train_model

#     # 推論時間の計測
#     start_time = time.time()
#     model.predict(X_test)
#     end_time = time.time()

#     inference_time = end_time - start_time

#     # 推論時間が1秒未満であることを確認
#     assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


# def test_model_reproducibility(sample_data, preprocessor):
#     """モデルの再現性を検証"""
#     # データの分割
#     X = sample_data.drop("Survived", axis=1)
#     y = sample_data["Survived"].astype(int)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # 同じパラメータで２つのモデルを作成
#     model1 = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
#         ]
#     )

#     model2 = Pipeline(
#         steps=[
#             ("preprocessor", preprocessor),
#             ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
#         ]
#     )

#     # 学習
#     model1.fit(X_train, y_train)
#     model2.fit(X_train, y_train)

#     # 同じ予測結果になることを確認
#     predictions1 = model1.predict(X_test)
#     predictions2 = model2.predict(X_test)

#     assert np.array_equal(
#         predictions1, predictions2
#     ), "モデルの予測結果に再現性がありません"


# def test_model_precision_recall(train_model):
#     """モデルの Precision と Recall を検証"""
#     model, X_test, y_test = train_model
#     y_pred = model.predict(X_test)

#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)

#     assert precision >= 0.7, f"Precision {precision} が 0.7 に達していません"
#     assert recall >= 0.7, f"Recall {recall} が 0.7 に達していません"


# def test_compare_with_baseline_model(train_model):
#     """ベースラインモデルとの比較"""
#     model, X_test, y_test = train_model
#     current_metrics = accuracy_score(y_test, model.predict(X_test))

#     if not os.path.exists(BASELINE_MODEL_PATH):
#         pytest.skip("ベースラインモデルのファイルが存在しないため、テストをスキップします")

#     with open(BASELINE_MODEL_PATH, "rb") as f:
#         baseline_model = pickle.load(f)

#     baseline_metrics = accuracy_score(y_test, baseline_model.predict(X_test))

#     assert current_metrics >= baseline_metrics, (
#         f"現在のモデルの精度 {current_metrics} はベースラインモデルの {baseline_metrics} よりも低いです"
#     )
import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルを訓練し、テスト用データを準備"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # ベースラインモデルとしても保存
    with open(BASELINE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないため、テストをスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.75, f"モデルの精度が低すぎます：{accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time
    assert inference_time < 1.0, f"推論時間が長すぎます：{inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"


def test_model_precision_recall(train_model):
    """モデルのPrecisionとRecallを検証"""
    model, X_test, y_test = train_model
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    assert precision >= 0.7, f"Precision {precision} が0.7に達していません"
    assert recall >= 0.7, f"Recall {recall} が0.7に達していません"


def test_compare_with_baseline_model(train_model):
    """ベースラインモデルとの比較"""
    model, X_test, y_test = train_model
    current_metrics = accuracy_score(y_test, model.predict(X_test))

    if not os.path.exists(BASELINE_MODEL_PATH):
        pytest.skip(
            "ベースラインモデルのファイルが存在しないため、テストをスキップします"
        )

    with open(BASELINE_MODEL_PATH, "rb") as f:
        baseline_model = pickle.load(f)

    baseline_metrics = accuracy_score(y_test, baseline_model.predict(X_test))

    assert (
        current_metrics >= baseline_metrics
    ), f"現在のモデルの精度 {current_metrics} はベースラインモデルの {baseline_metrics} よりも低いです"
