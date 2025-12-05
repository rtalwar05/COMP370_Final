import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("all_articles.csv")

TOPIC_COL = "Category"
SENT_COL = "Sentiment"

df[SENT_COL] = df[SENT_COL].astype(str).str.strip().str.capitalize()

ct = pd.crosstab(df[TOPIC_COL], df[SENT_COL])
ct_norm = ct.div(ct.sum(axis=1), axis=0)

sentiment_order = ["Negative", "Neutral", "Positive"]

sentiment_order = [s for s in sentiment_order if s in ct_norm.columns]
ct_norm = ct_norm[sentiment_order]

if "Negative" in ct_norm.columns:
    ct_norm = ct_norm.sort_values(by="Negative", ascending=False)

colors = [
    "#d62728",  
    "#ffeb3b",  
    "#2ca02c"   
][:len(sentiment_order)]  

ax = ct_norm.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=colors
)

ax.set_ylabel("Proportion of articles")
ax.set_xlabel("Topic")
ax.set_title("Sentiment distribution across topics")
ax.legend(title="Sentiment")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "topic_sentiment_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
