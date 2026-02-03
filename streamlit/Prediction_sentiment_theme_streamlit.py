import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import joblib
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer


st.title("Trustpilot Amazon Reviews")
st.sidebar.title("Sommaire")
pages = ["Exploration", "Interprétabilité", "Modélisation", "Saisir un avis"]
page = st.sidebar.radio("Aller vers", pages)


##############################################################
# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
STREAMLIT_DIR = BASE_DIR / "streamlit"
IMAGES_DIR = STREAMLIT_DIR / "images"

##############################################################
# =========================
# Chargement des modèles
# =========================
MODELS_DIR = BASE_DIR / "models"

@st.cache_resource
def load_models():
    # Sentence-BERT (chargé depuis Hugging Face)
    sbert_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # KMeans + labels (modèles entraînés)
    kmeans = joblib.load(MODELS_DIR / "kmeans_topics.pkl")
    cluster_labels = joblib.load(MODELS_DIR / "cluster_labels.pkl")

    # DistilBERT sentiment (modèle HF standard)
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = TFAutoModelForSequenceClassification.from_pretrained(
        sentiment_model_name
    )

    return (
        sbert_model,
        kmeans,
        cluster_labels,
        sentiment_model,
        sentiment_tokenizer
    )

sbert_model, kmeans, cluster_labels, sentiment_model, sentiment_tokenizer = load_models()

# =========================
# Fonction de prédiction
# =========================
def predict_review(review_text: str):

    review_text = clean_review_text(review_text)
    # ---- Sentiment ----
    enc = sentiment_tokenizer(
        review_text,
        truncation=True,
        padding="max_length",
        max_length=270,
        return_tensors="tf"
    )

    outputs = sentiment_model(enc, training=False)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]

    sentiment = "Positive" if np.argmax(probs) == 1 else "Negative"

    # ---- Topic ----
    embedding = np.asarray(
        sbert_model.encode([review_text])
    )
    cluster_id = int(kmeans.predict(embedding)[0])
    theme = cluster_labels[cluster_id]

    return {
        "sentiment": sentiment,
        "sentiment_score": float(np.max(probs)),
        "theme": theme
    }

# ==============================
# Fonction de nettoyage de texte
# ==============================
import re

def clean_review_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

##############################################################
# Chargement du dataset

@st.cache_data
def load_dataset():
    df = pd.read_csv(
        STREAMLIT_DIR / "train.csv",
        header=None,
        names=["label", "title", "text"]
    )
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df_negative = df[df["label"] == 1]
    df_positive = df[df["label"] == 2]
    return df, df_negative, df_positive


df, df_negative, df_positive = load_dataset()


#############################################################
#  PAGE 0 - EXPLORATION
#############################################################
if page == pages[0]:
    st.write("### Exploration")
    
    # Répartition des sentiments
    st.markdown("### 📊 Equilibre de la target")
    st.image(
        IMAGES_DIR / "repartition_sentiments.png",
        use_container_width=True
    )
    
    st.divider()

    # Boxplot longueur des avis
    st.markdown("### 📊 Longueur des avis")
    st.image(
        IMAGES_DIR / "boxplot_longueur_avis.png",
        use_container_width=True
    )

    st.divider()    

    # Boxplot longueur des avis par sentiment
    st.image(
        IMAGES_DIR / "boxplot_longueur_avis_par_sentiment.png",
        use_container_width=True
    )

    st.divider()    

    # Countplot longueur des avis
    st.image(
        IMAGES_DIR / "countplot_longueur_avis.png",
        use_container_width=True
    )

    st.divider()    

    # Countplot longueur des avis positifs
    st.image(
        IMAGES_DIR / "countplot_longueur_avis_positive.png",
        caption="Distribution longueur des avis positifs",
        use_container_width=True
    )

    st.divider()    

    # Countplot longueur des avis négatifs
    st.image(
        IMAGES_DIR / "countplot_longueur_avis_negative.png",
        caption="Distribution longueur des avis négatifs",
        use_container_width=True
    )

    st.divider()    

    # Violinplot longueur des avis positifs et negatifs
    st.image(
        IMAGES_DIR / "violinplot_longueur_avis.png",
        caption="Distribution longueur des avis",
        use_container_width=True
    )

    st.divider()    

    # WordCloud all words
    st.markdown("### 📊 Nuages de mots")
    st.image(
        IMAGES_DIR / "wordcloud.png",
        caption="Wordcloud – Corpus global",
        use_container_width=True
    )

    st.divider()    

    # WordCloud positive words
    st.image(
        IMAGES_DIR / "wordcloud_positive.png",
        caption="Wordcloud – Positive reviews",
        use_container_width=True
    )

    st.divider()    

    # WordCloud negative words
    st.image(
        IMAGES_DIR / "wordcloud_negative.png",
        caption="Wordcloud – Negative reviews",
        use_container_width=True
    )

    st.divider()    

    # Barplot trigrams positive words
    st.markdown("### 📊 Trigrammes")
    st.image(
        IMAGES_DIR / "barplot_trigrams_positive.png",
        caption="Top trigrams positive words",
        use_container_width=True
    )

    st.divider()    

    # Barplot trigrams negative words
    st.image(
        IMAGES_DIR / "barplot_trigrams_negative.png",
        caption="Top trigrams negative words",
        use_container_width=True
    )

#############################################################
#  PAGE 1 - Interprétabilité
#############################################################
if page == pages[1]:
    st.write("### Interprétabilité")

    # SVM linear interpretability
    st.markdown("### 🔍 Interprétabilité SVM Linear (coefficients)")
    st.image(
        IMAGES_DIR / "interpretability_svm.png",
        use_container_width=True
    )

    st.divider()    

    # Random Forest interpretability
    st.markdown("### 🔍 Interprétabilité Random Forest (RFE)")
    st.image(
        IMAGES_DIR / "interpretability_random_forest.png",
        use_container_width=True
    )

    st.divider()    

    # XGBoost interpretability
    st.markdown("### 🔍 Interprétabilité XGBoost (feature importance)")    
    st.image(
        IMAGES_DIR / "interpretability_xgboost.png",
        use_container_width=True
    )

    st.divider()    

    # DistilBERT interpretability
    st.markdown("### 🔍 Interprétabilité DistillBERT (LIME)") 
    st.image(
        IMAGES_DIR / "interpretability_distilbert.png",
        use_container_width=True
    )


#############################################################
#  PAGE 2 - Modélisation
#############################################################
if page == pages[2]:
    st.write("### Modélisation")
    st.markdown(f"###  Prédictions de quelques avis du dataset de test")


    # =========================
    # Test de reviews du dataset
    # =========================
    random_state = 43

    def safe_sample(df, n, random_state):
        if len(df) == 0:
            return pd.DataFrame(columns=df.columns)
        return df.sample(
            n=min(n, len(df)),
            random_state=random_state
        )

    neg_samples = safe_sample(df_negative, 5, random_state)[["text", "label"]]
    pos_samples = safe_sample(df_positive, 5, random_state)[["text", "label"]]

    test_df = (
        pd.concat([neg_samples, pos_samples])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    label_mapping = {
        1: "Negative",
        2: "Positive"
    }

    for i, row in test_df.iterrows():
        review_text = row["text"]
        true_label = label_mapping[row["label"]]

        result = predict_review(review_text)

        st.markdown(f"### 📝 Avis {i+1}")
        st.info(review_text)

        st.write(f"→ Sentiment réel   : **{true_label}**")
        st.write(
            f"→ Sentiment prédit : **{result['sentiment']}** "
            f"(score = {result['sentiment_score']:.3f})"
        )
        st.write(f"→ Thème prédit     : **{result['theme']}**")

        st.divider()

#############################################################
#  PAGE 3 - Saisie Avis
#############################################################
if page == pages[3]:

    # =========================
    # Test 1 review personnalisée
    # =========================

    st.markdown("### ✍️ Saisissez un avis")

    review = st.text_area(
        label="(Ctrl+Entrée) pour valider",
        placeholder="Ex : The movie stopped working after two weeks and feels very cheap.",
        height=200
    )

    # On ne lance la prédiction que si quelque chose est saisi
    if review.strip():
        result = predict_review(review)

        # titre centré
        st.markdown("<h3 style='text-align: left; margin-left: 10%;'>🔮 Predictions :</h3>",unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                f"**Sentiment** : {result['sentiment']}  \n"
                f"<span style='color:gray'>(score = {result['sentiment_score']:.3f})</span>",
                unsafe_allow_html=True
            )
            # ici plus tard : st.image("smiley_positive.png") par ex.

        with col2:
            st.markdown(
                f"**Thème** : {result['theme']}"
            )
            #  ici plus tard : st.image("logo_topic.png")
    

    