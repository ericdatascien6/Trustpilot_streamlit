import streamlit as st
import pandas as pd
import numpy as np


st.title("Projet Amazon Trustpilot")
st.sidebar.title("Sommaire")
pages=["Exploration", "Preprocessing", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)


##############################################################
# Chargement du dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv(
        "train.csv",
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
        "images/repartition_sentiments.png",
        caption="Repartition des sentiments",
        use_container_width=True
    )
    # WordCloud
    st.image(
        "images/wordcloud.png",
        caption="Wordcloud – Corpus global",
        use_container_width=True
    )





#############################################################
if page == pages[1] : 
    st.write("### Preprocessing")



#############################################################
if page == pages[2] : 
    st.write("### Modélisation")

    ###########################################################################
    #   Prédiction du sentiment et du thème - 
    ###########################################################################
    import numpy as np
    import tensorflow as tf
    import joblib
    from transformers import (
        TFDistilBertForSequenceClassification,
        DistilBertTokenizerFast
    )
    from sentence_transformers import SentenceTransformer



    # =========================
    # Rechargement modèles
    # =========================
    @st.cache_resource
    def load_models():
        sbert_model = SentenceTransformer("../models/sentence_bert")
        kmeans = joblib.load("../models/kmeans_topics.pkl")
        cluster_labels = joblib.load("../models/cluster_labels.pkl")

        sentiment_model = TFDistilBertForSequenceClassification.from_pretrained(
            "../models/distilbert_sentiment"
        )
        sentiment_tokenizer = DistilBertTokenizerFast.from_pretrained(
            "../models/distilbert_sentiment"
        )

        return sbert_model, kmeans, cluster_labels, sentiment_model, sentiment_tokenizer
    
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
    # Test 1 reviews personnalisée
    # =========================
    review = "The movie stopped working after two weeks and feels very cheap."

    result = predict_review(review)

    st.markdown(f"### 📝 Avis rédigé")
    st.success(review)
    st.write(f"Thème :", result["theme"])
    st.write(f"Sentiment : {result['sentiment']}  ---   (Score = {result['sentiment_score']})")
    
    

    # =========================
    # Test de reviews du dataset
    # =========================
    # Sélection aléatoire de 5 avis positifs et 5 avis négatifs
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


    # Dataset de test final (mélangé)
    test_df = (
        pd.concat([neg_samples, pos_samples])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    # Correspondance labels
    label_mapping = {
        1: "Negative",
        2: "Positive"
    }

    # Prédictions
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

