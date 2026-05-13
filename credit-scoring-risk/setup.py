from setuptools import find_packages, setup

setup(
    name="credit-scoring",
    version="0.1.0",
    description="Credit scoring and risk segmentation with XGBoost, LightGBM, and KMeans",
    author="Sebastián Segura",
    author_email="juan.ssr@hotmail.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "lightgbm>=4.3",
        "imbalanced-learn>=0.12",
        "optuna>=3.6",
        "shap>=0.45",
        "fastapi>=0.111",
        "uvicorn[standard]>=0.29",
        "pydantic>=2.7",
        "mlflow>=2.13",
        "joblib>=1.4",
        "openpyxl>=3.1",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "ruff", "httpx"],
    },
    entry_points={
        "console_scripts": [
            "credit-train=train:main",
            "credit-serve=api.app:start",
        ],
    },
)
