# Wine Quality Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline implemented in Python to analyze and predict red wine quality based on physicochemical characteristics. This system transforms structural chemical data into an automated classification tool using an ensemble learning approach.

---

## 📌 Table of Contents
- [Business Problem Statement](#-business-problem-statement)
- [Workflow Architecture](#-workflow-architecture)
- [Dataset Overview](#-dataset-overview)
- [Data Analysis & Key Insights](#-data-analysis--key-insights)
- [Data Preprocessing & Binarization](#-data-preprocessing--binarization)
- [Model Selection & Ensemble Mechanics](#-model-selection--ensemble-mechanics)
- [Evaluation Metrics](#-evaluation-metrics)
- [Installation & Usage](#-installation--usage)

---

## 🏢 Business Problem Statement

In the commercial wine manufacturing industry, determining product quality traditionally relies on human organoleptic evaluation (sensory tasting by experts). This process is highly subjective, expensive, resource-intensive, and prone to inconsistency. 

**The Solution:** This project replaces or augments sensory panels with automated predictive analytics. By feeding physical and chemical characteristics into a trained machine learning system, manufacturing plants can dynamically audit batch quality during production stages, optimizing raw materials and ensuring standard profile consistencies.

---



## 🔄 Workflow Architecture

The execution pipeline follows a rigorous data science engineering workflow:
[ Raw Dataset ] ➔ [ Exploratory Data Analysis (EDA) ] ➔ [ Data Preprocessing ]
│
[ Model Evaluation ] 🗲 [ Random Forest Training ] 🧱 [ Train-Test Split (80:20) ]
│
[ Production-Ready Predictive System ]


1. **Data Collection:** Importing a structural physicochemical dataset from Kaggle.
2. **Exploratory Data Analysis (EDA):** Generating statistical summaries and parsing distribution relationships using advanced plotting.
3. **Data Preprocessing:** Splitting parameters from targeted vectors, running handling validations for empty values, and executing target label binarization.
4. **Train-Test Split:** Segregating the dataset matrices into an 80% distribution for algorithm training and a 20% holdout for metric benchmarking.
5. **Model Training:** Fitting data across an ensemble of randomized decision boundaries.
6. **Evaluation:** Validating predictive confidence against true unexposed markers.
7. **Predictive System:** Compiling a processing wrapper for inference execution on single-row parameters.

---

## 📊 Dataset Overview

The system uses the **Red Wine Quality Dataset** consisting of **1,599 instances** and **12 features**. Each data entry represents a specific red wine batch profile.

### Physicochemical Features:
* `fixed acidity`: Primary organic acids found in grapes (tartaric/malic).
* `volatile acidity`: The amount of acetic acid in wine, which at high levels leads to an unpleasant, vinegar-like taste.
* `citric acid`: Found in small quantities; adds freshness and flavor characteristics.
* `residual sugar`: The amount of sugar remaining after fermentation stops.
* `chlorides`: The amount of salt in the wine.
* `free sulfur dioxide`: Prevents microbial growth and the oxidation of wine.
* `total sulfur dioxide`: Total amount of free and bound forms of $SO_2$.
* `density`: Mass per unit volume, varying based on alcohol and sugar content.
* `pH`: Describes how acidic or basic the wine is on a scale from 0 (very acidic) to 14 (very basic).
* `sulphates`: A wine additive that acts as an antimicrobial and antioxidant.
* `alcohol`: The percentage of alcohol content by volume.
* `quality`: The target output score assessed by sensory experts (originally graded from 3 to 8).

---

## 🔍 Data Analysis & Key Insights

Before model fitting, structural analysis and visualizations (`seaborn` and `matplotlib`) were executed to discover underlying feature mechanics:

* **Missing Values Check:** Evaluated via `.isnull().sum()`. The dataset contains **0 missing values**, eliminating the need for mean imputation or row deletion.
* **Target Distribution:** A Categorical Count Plot (`sns.catplot`) highlighted that the target variable is highly concentrated around quality scores 5 and 6, showing an implicit skewness in regional expert grading.
* **Volatile Acidity Correlation:** Bar plot trends (`sns.barplot`) prove an **inverse proportion** between volatile acidity and quality. Higher acetic concentrations explicitly correspond to poor quality grades (scores 3–4).
* **Citric Acid Correlation:** Conversely, increased citric acid levels map directly onto higher expert ratings, marking it as a significant structural variable.
* **Correlation Heatmap Matrix:** A comprehensive matrix plot (`sns.heatmap`) using a single-point floating configuration reveals that **Alcohol Content** exhibits the highest positive linear correlation score (~0.5) relative to target quality.

---

## ⚙️ Data Preprocessing & Binarization

### 1. Feature-Label Isolation
The target vector `quality` is removed from the input matrix using a column axis drop operation (`axis=1`), creating an independent variable space ($X$) and a dependent space ($Y$).

### 2. Label Binarization (Why?)
A multiclass prediction structure (predicting precise integers between 3 and 8) introduces a high degree of variance given the 1,599 data point constraint. To turn this into a high-precision, business-focused decision model, the targets undergo binary encoding via a Python `lambda` map wrapper:

$$\text{Encoded Quality} = \begin{cases} 1 \quad (\text{Good/Premium Quality}), & \text{if } \text{quality} \geq 7 \\ 0 \quad (\text{Bad/Standard Quality}), & \text{if } \text{quality} < 7 \end{cases}$$

This binarization translates raw data into clear, actionable commercial logic: **Is this batch premium grade ($1$), or standard/substandard ($0$)?**

### 3. Holdout Stratification
The dataset is segmented via `train_test_split` with a designated distribution constraint (`test_size=0.2`) and an explicit `random_state=2` for analytical reproducibility:
* **Training Set ($X_{train}, Y_{train}$):** 1,279 samples (used to map patterns).
* **Testing Set ($X_{test}, Y_{test}$):** 320 samples (isolated to test generalization limits).

---

## 🌲 Model Selection & Ensemble Mechanics

This pipeline utilizes a **Random Forest Classifier**, an advanced ensemble learning algorithm. 

### Why Random Forest?
A single Decision Tree splits features based on information gain, but it is highly susceptible to overfitting and structural instability. A Random Forest constructs a "forest" of independent decision trees via a process called **Bootstrap Aggregating (Bagging)**.

During the fitting cycle:
1. Each tree is trained on a random sample of rows and a random subset of attributes.
2. Individual trees arrive at isolated classification results.
3. The ensemble aggregates these individual trees, executing a **majority voting protocol** to determine the final output class.

This mitigates variance, handles potential nonlinear decision boundaries effortlessly, and provides high predictive stability out of the box.

---

## 📈 Evaluation Metrics

The pipeline's performance is gauged using Scikit-Learn's structural `accuracy_score` metric, comparing predictions against actual isolated testing values:

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples Evaluated}}$$

* **Final Achieved Testing Accuracy:** **`92.5%`** (approximating `93%`).
* **Significance:** An accuracy threshold exceeding $80\%$ indicates high reliability. Achieving $\sim93\%$ confirms that the ensemble's decision boundaries generalize exceptionally well to new data points.

---

## 💻 Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the necessary library dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
git clone https://github.com/Ved-Bhanushali03/Wine-Quality-Prediction-.git
cd Wine_Quality_Prediction
