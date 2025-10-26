# Iris Flower Classification Using Machine Learning 🌸

A comprehensive machine learning project to classify Iris flower species based on physical measurements. The project includes model training with multiple algorithms, performance evaluation, and a production-ready FastAPI REST API for real-time predictions.

---

## 📌 Project Overview

This project leverages machine learning algorithms to classify Iris flowers into three species based on their sepal and petal measurements. The dataset contains **150 samples** of Iris flowers, representing one of the most famous datasets in machine learning and pattern recognition.

- Multiple ML models trained and compared using **scikit-learn**
- Comprehensive **model evaluation** with cross-validation
- Deployed as a **RESTful API** using FastAPI
- Real-time predictions with confidence scores
- Professional error handling and input validation

---

## 🚀 Features

- **Machine Learning Pipeline**: 
  - Data preprocessing with duplicate and null value handling
  - Label encoding for target classes
  - Train-test split with stratification
  - Cross-validation for robust performance estimation

- **Model Training & Comparison**: 
  - Random Forest Classifier
  - Logistic Regression with scaling
  - Decision Tree Classifier
  - Support Vector Machine (SVM) with RBF kernel
  - Automatic best model selection based on accuracy

- **REST API with FastAPI**: 
  - `/` - Health check endpoint with status monitoring
  - `/predict` - Single flower species prediction
  - `/predict/batch` - Batch prediction for multiple samples
  - `/model-info` - Model metadata and configuration
  - `/species` - Available species and encoding information

- **Production-Ready Features**:
  - Pydantic models for input validation
  - Comprehensive error handling
  - Confidence scores and probability distributions
  - Interactive API documentation (Swagger UI)
  - Model persistence with joblib

---

## 🛠️ Technologies Used

- **Machine Learning**: Random Forest, Logistic Regression, Decision Tree, SVM (`scikit-learn`)
- **Programming Language**: Python 3.8+
- **Libraries**:
  - `scikit-learn`: For ML algorithms and preprocessing
  - `pandas`: For data manipulation
  - `numpy`: For numerical operations
  - `joblib`: For model serialization
  - `fastapi`: For REST API development
  - `uvicorn`: ASGI server for FastAPI
  - `pydantic`: For data validation
- **Development Environment**: Google Colab (model training), Local development (API)
- **API Framework**: FastAPI with automatic OpenAPI documentation

---

## 📂 Dataset

- **Filename**: `Iris.csv`
- **Source**: UCI Machine Learning Repository / Kaggle
- **Description**: Contains measurements of Iris flowers from three different species
- **Size**: 150 samples with 5 features (4 input features + 1 target)
- **Features**:
  - **SepalLengthCm**: Sepal length in centimeters
  - **SepalWidthCm**: Sepal width in centimeters
  - **PetalLengthCm**: Petal length in centimeters
  - **PetalWidthCm**: Petal width in centimeters
  - **Species**: Target variable (Iris-setosa, Iris-versicolor, Iris-virginica)

---

## 📊 Best Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 96.67%
- **Cross-Validation Accuracy**: 95.83%
- **Training/Test Split**: 80% / 20%
- **Key Strengths**:
  - Robust to overfitting
  - Handles non-linear relationships
  - Provides feature importance
  - High prediction confidence

### Model Comparison Results

| Model | Cross-Val Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| Random Forest | 95.83% | 96.67% |
| Logistic Regression | 95.00% | 96.67% |
| Decision Tree | 95.00% | 93.33% |
| SVM | 96.67% | 96.67% |

---

## 📁 Project Structure

```
iris-flower-classification/
│
├── Backend/
│   ├── app.py                      # FastAPI application
│   └── best_model.joblib           # Trained ML model
│
├── iris_flower_classification.py   # Model training script
├── Iris.csv                        # Dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Kapil-MLDev/Iris-Flower-Classification-.git
cd iris-flower-classification
```

### Step 2: Create Virtual Environment
```bash
# Using conda
conda create -n Ml_models python=3.9+
conda activate Ml_models

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```
pandas
numpy
scikit-learn
fastapi
uvicorn
pydantic
joblib
```

### Step 4: Train the Model (Optional)
If you want to retrain the model:
```bash
python iris_flower_classification.py
```
This will generate `best_model.joblib` in your working directory.

### Step 5: Move Model to Backend
```bash
# Windows
copy best_model.joblib Backend\best_model.joblib

# macOS/Linux
cp best_model.joblib Backend/best_model.joblib
```

---

## ▶️ How to Run the API

### Start the FastAPI Server

```bash
cd Backend
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will start at `http://localhost:8000`

### Access Interactive Documentation

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---
---

## 📈 Key Insights from Data Analysis

- **Perfect Separation**: Iris-setosa is linearly separable from other species
- **Feature Importance**: Petal length and petal width are the most discriminative features
- **Species Characteristics**:
  - **Iris-setosa**: Smallest petals (1.0-1.9 cm length)
  - **Iris-versicolor**: Medium-sized petals (3.0-5.1 cm length)
  - **Iris-virginica**: Largest petals (4.5-6.9 cm length)
- **Data Quality**: No missing values, perfectly balanced classes (50 samples each)
- **Model Performance**: All models achieved >93% accuracy, indicating clear separability
---
## 👤 Author

**S.Kapila Deshapriya**
- LinkedIn: [https://www.linkedin.com/in/kapila-Deshapriya/]

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Iris dataset
- FastAPI team for the excellent web framework
- scikit-learn developers for comprehensive ML tools
- The open-source community for inspiration and support

---

## 📚 References

- [Iris Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Fisher, R.A. (1936). "The use of multiple measurements in taxonomic problems"](https://en.wikipedia.org/wiki/Iris_flower_data_set)

---

**Made with ❤️ and Python**

*"From botanical classification to modern machine learning - A journey through the famous Iris dataset 🌸"*
