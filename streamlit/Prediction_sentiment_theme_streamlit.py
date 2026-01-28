import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


st.title("Projet Amazon Trustpilot")
st.sidebar.title("Sommaire")
pages = ["Exploration", "Preprocessing", "Modélisation"]
page = st.sidebar.radio("Aller vers", pages)


##############################################################
# Chemins robustes
BASE_DIR = Path(__file__).resolve().parent.parent
STREAMLIT_DIR = BASE_DIR / "streamlit"
IMAGES_DIR = STREAMLIT_DIR / "images"


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


##############################################################
if page == pages[0]:

    # Répartition des sentiments
    st.image(
        IMAGES_DIR / "repartition_sentiments.png",
        caption="Répartition des sentiments",
        use_container_width=True
    )

    # WordCloud
    st.image(
        IMAGES_DIR / "wordcloud.png",
        caption="Wordcloud – Corpus global",
        use_container_width=True
    )


#############################################################
if page == pages[1]:
    st.write("### Preprocessing")


#############################################################
if page == pages[2]:
    st.write("### Modélisation")

    ###########################################################################
    #   Prédiction du sentiment et du thème
    ###########################################################################
    import tensorflow as tf
    import joblib
    from transformers import (
        AutoTokenizer,
        TFAutoModelForSequenceClassification
    )
    from sentence_transformers import SentenceTransformer


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

        # ---- Sentiment ----
        enc = sentiment_tokenizer(
            review_text,
            truncation=True,
            padding="max_length",
            max_length=256,
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


    # =========================
    # Test 1 review personnalisée
    # =========================
    review = "The movie stopped working after two weeks and feels very cheap."

    result = predict_review(review)

    st.markdown("### 📝 Avis rédigé")
    st.success(review)
    st.write("Thème :", result["theme"])
    st.write(
        f"Sentiment : {result['sentiment']}  ---   "
        f"(Score = {result['sentiment_score']:.3f})"
    )


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
