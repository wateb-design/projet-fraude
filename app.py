import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Détection de Fraude – CNN", page_icon="🏦", layout="wide")

@st.cache_data
def load_data():
    with open("metriques.json") as f:
        metriques = json.load(f)
    cm       = np.load("confusion_matrix.npy")
    fpr      = np.load("fpr.npy")
    tpr      = np.load("tpr.npy")
    loss     = np.load("history_loss.npy")
    val_loss = np.load("history_val_loss.npy")
    acc      = np.load("history_acc.npy")
    val_acc  = np.load("history_val_acc.npy")
    return metriques, cm, fpr, tpr, loss, val_loss, acc, val_acc

metriques, cm, fpr, tpr, loss, val_loss, acc, val_acc = load_data()

st.title("🏦 Détection de Fraude Bancaire par CNN")
st.divider()

st.header("📊 Performance du modèle")
col1,col2,col3,col4 = st.columns(4)
col1.metric("Accuracy", f"{metriques['accuracy']:.2%}")
col2.metric("F1 Score", f"{metriques['f1']:.2%}")
col3.metric("ROC AUC",  f"{metriques['roc_auc']:.2%}")
col4.metric("Recall",   f"{metriques['recall']:.2%}")
st.divider()

st.header("🔥 Matrice de Confusion")
fig1, ax1 = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
            xticklabels=["Normal","Fraude"],
            yticklabels=["Normal","Fraude"])
ax1.set_ylabel("Réel"); ax1.set_xlabel("Prédit")
st.pyplot(fig1)
st.divider()

st.header("📈 Courbes ROC & Entraînement")
c1, c2, c3 = st.columns(3)
with c1:
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC={metriques['roc_auc']:.3f}")
    ax2.plot([0,1],[0,1],'k--')
    ax2.set_title("Courbe ROC")
    ax2.legend()
    st.pyplot(fig2)
with c2:
    fig3, ax3 = plt.subplots()
    ax3.plot(loss, label="Train")
    ax3.plot(val_loss, label="Val")
    ax3.set_title("Loss")
    ax3.legend()
    st.pyplot(fig3)
with c3:
    fig4, ax4 = plt.subplots()
    ax4.plot(acc, label="Train")
    ax4.plot(val_acc, label="Val")
    ax4.set_title("Accuracy")
    ax4.legend()
    st.pyplot(fig4)
