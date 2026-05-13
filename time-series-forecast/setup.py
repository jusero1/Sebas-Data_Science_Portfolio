from setuptools import find_packages, setup

setup(
    name="ts-forecast",
    version="0.1.0",
    description="LSTM/GRU time series forecasting for household electric power consumption",
    author="Sebastián",
    author_email="zebas.sr@gmail.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "tensorflow>=2.13",
        "scikit-learn>=1.4",
        "fastapi>=0.111",
        "uvicorn[standard]>=0.29",
        "pydantic>=2.7",
        "mlflow>=2.13",
        "joblib>=1.4",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "ruff", "httpx"],
    },
    entry_points={
        "console_scripts": [
            "ts-train=train:main",
            "ts-serve=api.app:start",
        ],
    },
)
