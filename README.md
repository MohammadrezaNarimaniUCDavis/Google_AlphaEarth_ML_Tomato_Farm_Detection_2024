# Tomato Field Detection in California Using Google AlphaEarth Satellite Embeddings and Machine Learning

**A machine learning workflow to identify tomato fields in California using LandIQ 2024 field boundaries and Google AlphaEarth satellite embeddings.**

---

## Poster Showcase 🎯

Click the image below to view the **full high-resolution PDF** of the project poster.

[![Project Poster](Poster_Showcase/ESEARCH_Spring_2026_Poster.png)](Poster_Showcase/ESEARCH_Spring_2026_Poster.pdf)

---

## 📖 Overview

The goal of this project was to detect tomato fields across California using field boundary data, satellite foundation-model embeddings, and machine learning. 

This project developed a machine learning workflow to identify tomato fields in California using LandIQ 2024 field boundaries and Google AlphaEarth satellite embeddings. Tomato and non-tomato field polygons were selected from LandIQ, AlphaEarth embeddings were clipped to each field, and each field was summarized as a 64-dimensional feature vector. Three base classifiers—Random Forest, Naive Bayes, and Linear SVM—were trained independently, and their predictions were combined using a logistic regression stacking model. Results showed high classification performance, with Linear SVM achieving the best standalone accuracy and F1 score, and the stacked ensemble achieving the highest ROC-AUC. The project demonstrates the potential of satellite foundation-model embeddings for rapid crop identification at field scale.

---

## 🎯 Motivation

Field-level crop maps are very useful for agricultural monitoring, crop inventory, resource management, and decision-making. However, official crop maps and ground-labeled datasets can take time to produce, especially at large spatial scales.

Because of this delay, the project explored whether satellite foundation-model embeddings could help identify crop types faster. Specifically, the project focused on detecting tomato fields in California using Google AlphaEarth satellite embeddings and machine learning.

The main idea was: Instead of waiting for delayed crop maps or relying heavily on manual ground labeling, we tested whether AlphaEarth embeddings could provide a fast and scalable way to classify fields as tomato or non-tomato.

---

## 📊 Data Sources

The project used two main data sources:

1. **LandIQ 2024 Field Shapefiles**  
   We used LandIQ 2024 shapefiles for California agricultural field boundaries. These shapefiles contain field polygons and crop-type information for farms across California.

2. **Google AlphaEarth Satellite Embeddings**  
   We used the Google Satellite Embedding Dataset / AlphaEarth embeddings. AlphaEarth embeddings combine information from multiple remote-sensing sources. These sources include optical satellite imagery, radar, and LiDAR. They also capture other contextual signals from the landscape. This information is compressed into 64 compact features for each field. This made AlphaEarth suitable for machine learning because each field could be represented as a clean numerical feature vector.

---

## 🛠️ Dataset Preparation & Feature Extraction

The dataset preparation followed these steps:

1. **Load LandIQ 2024 shapefiles:** We started with LandIQ 2024 shapefiles containing California field boundaries and crop labels.
2. **Separate tomato fields:** We filtered the LandIQ dataset to select the polygons labeled as tomato fields. This produced approximately 4,700 tomato field polygons.
3. **Select non-tomato fields:** We selected approximately 4,700 non-tomato field polygons from all other crop classes. This created a balanced tomato vs. non-tomato classification dataset.
4. **Clip AlphaEarth imagery to field polygons:** For each selected field polygon, we clipped the AlphaEarth satellite embedding data to the field boundary.
5. **Compute field-level embedding features:** For each clipped field, we summarized the AlphaEarth data by computing the mean value of each embedding band across the field. This produced one 64-dimensional feature vector per field.
6. **Create final machine learning table:** The final dataset contained one row per field, 64 AlphaEarth embedding features per field, and a class label (Tomato or Non-tomato). For model testing and visualization, a balanced subsample of 500 tomato fields and 500 non-tomato fields was used.

---

## 🔍 Exploratory Analysis

Before training models, we explored the 64-dimensional embeddings:

- **PCA Visualization:** We used PCA to reduce the 64-dimensional AlphaEarth embeddings into two principal components. This allowed us to visualize whether tomato and non-tomato fields showed separation in embedding space. The PCA plot showed that tomato and non-tomato fields had meaningful separation, suggesting that the AlphaEarth embeddings contained useful information for distinguishing crop types.
- **Mean Embedding Profile by Class:** We also compared the average AlphaEarth embedding values across the 64 bands for tomato and non-tomato fields. The class-wise mean profiles were similar overall, but they differed enough across certain bands to support accurate classification.

---

## 🤖 Machine Learning Models

The project used three independent base machine learning models and one stacked ensemble model.

### Base Models
Each model was trained to classify fields as tomato or non-tomato using the 64 AlphaEarth embedding features:
- Random Forest
- Naive Bayes
- Linear Support Vector Machine (SVM)

### Stacked Ensemble Model
After training the base models, we used a logistic regression meta-classifier as a stacking model to combine the strengths of the individual models and create a more consensus-based classifier.
Workflow:
1. Train Random Forest, Naive Bayes, and Linear SVM.
2. Generate prediction probabilities from each base model.
3. Feed those predictions into a logistic regression model.
4. Use the logistic regression model as the final meta-classifier.

---

## 📈 Results

The models were evaluated on a held-out test set using common classification metrics:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Random Forest | 0.960 | 0.979 | 0.940 | 0.959 | 0.994 |
| Naive Bayes | 0.940 | 0.978 | 0.900 | 0.938 | 0.992 |
| SVM (linear) | 0.965 | 0.970 | 0.960 | 0.965 | 0.995 |
| Stacked (Meta) | 0.955 | 0.979 | 0.930 | 0.954 | 0.997 |

The confusion matrices showed that all models performed well, with most samples correctly classified. 

---

## 🌟 Feature Importance

To understand which AlphaEarth embedding bands mattered most, we used Random Forest Gini importance. The top 5 important AlphaEarth bands were:
- **A40** — importance 0.15
- **A10** — importance 0.11
- **A29** — importance 0.08
- **A48** — importance 0.06
- **A47** — importance 0.05

The stacking weights from the logistic regression meta-classifier were:
- Linear SVM — 4.15
- Random Forest — 2.74
- Naive Bayes — 1.71

This means the stacked model relied most heavily on the Linear SVM predictions, which agrees with the model performance results where Linear SVM was the best standalone model.

---

## 💡 Key Findings

- AlphaEarth embeddings can distinguish tomato fields from non-tomato fields in California.
- The Linear SVM model achieved the best standalone classification performance.
- The stacked ensemble achieved the highest ROC-AUC, showing strong consensus-based discrimination.
- The PCA visualization showed meaningful separation between tomato and non-tomato fields.
- Feature importance analysis showed that certain AlphaEarth embedding bands were especially useful.
- This approach could help reduce dependence on delayed crop maps and manual ground labeling.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MohammadrezaNarimaniUCDavis/Google_AlphaEarth_ML_Tomato_Farm_Detection_2024.git
   cd Google_AlphaEarth_ML_Tomato_Farm_Detection_2024
   ```
2. Open and run the main notebook:
   - `Code/tomato_field_classifier_2024.ipynb`

---

## 👤 Authors

- **Yaohong Chen**
- **Mohammadreza Narimani** 
