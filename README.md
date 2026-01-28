# Projet Amazon Trustpilot — NLP, Modélisation & Streamlit

## 🔗 Application en ligne
👉 Accéder à l’application Streamlit : 
https://trustpilotapp-aeaia97rx7piuhlppj7aun.streamlit.app/ 


Cette application permet d’explorer, analyser et prédire automatiquement :
- le **sentiment** d’un avis client (positif / négatif)
- la **thématique principale** associée à l’avis
- des **insights métier** exploitables via une interface interactive

---

## 🧭 Contexte & Objectifs

### Objectif du projet
Ce projet s’inscrit dans un cas d’usage réaliste : **Trustpilot** souhaite fournir à ses entreprises clientes un module d’analyse automatique des avis, capable de :

- classifier le **sentiment** (positif / négatif)
- extraire automatiquement les **grandes thématiques** présentes dans les retours clients
- synthétiser les insights dans un **tableau de bord métier**

Nous travaillons sur un **client fictif** :

> **Paul**, Responsable Marketing Produits Loisirs chez **Amazon**, utilisateur intensif de Trustpilot.

---

## 📊 Données utilisées

Les avis Trustpilot n’étant pas disponibles publiquement à grande échelle, ce projet repose sur un **dataset proxy robuste** :

### Amazon Reviews Polarity (Kaggle)
- **3,6 M** avis pour l’entraînement  
- **0,4 M** avis pour le test  
- **2 classes équilibrées** (positif / négatif)  
- Données textuelles riches : livres, films, musique, jeux vidéo…  

👉 Ce dataset est particulièrement adapté pour simuler un **usage Trustpilot haute volumétrie**.

---

## 🧪 Travail réalisé dans ce dépôt

### 1 — Exploration & Data Visualisation
**Notebook :** `Exploration_dataviz.ipynb`

- Analyse exploratoire des données
- Visualisations statistiques
- Étude des distributions textuelles
- Premiers insights métier

---

### 2 — Rapport de projet
**Fichier :** `Rendu3.pdf`

Ce document constitue un livrable complet contenant :
- Contexte & vision produit (aligné avec le **Product Vision Board**)
- Description détaillée du dataset
- Analyse univariée & textuelle
- Étude de la qualité linguistique
- Synthèse des insights métiers
- Définition du pipeline de pré-processing
- Modélisation

---

## 🤖 Modélisation & NLP

Le projet repose sur une architecture NLP moderne :

- **Analyse de sentiment**
  - DistilBERT (Transformers – Hugging Face)
  - Classification binaire (positif / négatif)

- **Extraction de thématiques**
  - Sentence-BERT pour les embeddings sémantiques
  - Clustering KMeans
  - Interprétation métier via labels de clusters

Les modèles sont chargés dynamiquement dans l’application Streamlit.

---

## 🖥 Application Streamlit

L’application Streamlit permet :
- la visualisation des analyses exploratoires
- la prédiction du sentiment et du thème d’un avis personnalisé
- le test sur des avis issus du dataset

### Déploiement
- Application déployée sur **Streamlit Cloud**
- Code source versionné sur **GitHub**
- Les modèles volumineux ne sont pas versionnés (contraintes GitHub)

---

## 🛠 Stack technique

- Python
- Streamlit
- Pandas / NumPy
- TensorFlow
- Hugging Face Transformers
- Sentence-Transformers
- Scikit-learn
- Joblib

---

## 📁 Structure du projet

trustpilot_streamlit/
├── models/
│ ├── kmeans_topics.pkl
│ └── cluster_labels.pkl
├── streamlit/
│ ├── images/
│ ├── train.csv (échantillon)
│ └── Prediction_sentiment_theme_streamlit.py
├── requirements.txt
├── runtime.txt
└── README.md
