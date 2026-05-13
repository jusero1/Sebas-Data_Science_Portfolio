from setuptools import find_packages, setup

setup(
    name="cv-pneumonia",
    version="0.1.0",
    description="Chest X-Ray pneumonia detection with CNN and ResNet50 Transfer Learning",
    author="Sebastián",
    author_email="zebas.sr@gmail.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "Pillow>=10.3",
        "tensorflow>=2.13",
        "scikit-learn>=1.4",
        "fastapi>=0.111",
        "uvicorn[standard]>=0.29",
        "pydantic>=2.7",
        "python-multipart>=0.0.9",
        "mlflow>=2.13",
        "opencv-python-headless>=4.9",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "ruff", "httpx"],
    },
)
