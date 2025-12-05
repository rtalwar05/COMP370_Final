import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "all_articles.csv"

TOPIC_COL = "Category"
SENT_COL = "Sentiment"
DATE_COL = "publishedAt"

CUTOFF_DATE = pd.Timestamp(2025, 11, 4)  

df = pd.read_csv(CSV_PATH)
print("Columns in CSV:", df.columns)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce")
df = df.dropna(subset=[DATE_COL])
df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)  

df[SENT_COL] = (
    df[SENT_COL]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"negative": "Negative", "neutral": "Neutral", "positive": "Positive"})
)

df = df.dropna(subset=[SENT_COL])
print("Unique sentiments after normalization:", df[SENT_COL].unique())

df["Period"] = df[DATE_COL].apply(
    lambda d: "Pre-Nov 4th" if d < CUTOFF_DATE else "Post-Nov 4th"
)

print("\nArticle counts by period:")
print(df["Period"].value_counts())

ct = pd.crosstab(df["Period"], df[SENT_COL])
ct_norm = ct.div(ct.sum(axis=1), axis=0)  

print("\nRaw counts (Period × Sentiment):")
print(ct)
print("\nNormalized proportions (Period × Sentiment):")
print(ct_norm)

sentiment_order = ["Negative", "Neutral", "Positive"]
sentiment_order = [s for s in sentiment_order if s in ct_norm.columns]
ct_norm = ct_norm[sentiment_order]

period_order = ["Pre-Nov 4th", "Post-Nov 4th"]
ct_norm = ct_norm.reindex(period_order)

colors = [
    "#d62728",  
    "#ffeb3b",  
    "#2ca02c",  
][:len(sentiment_order)]

ax = ct_norm.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    color=colors
)

ax.set_ylabel("Proportion of articles")
ax.set_xlabel("Period")
ax.set_title("Sentiment Distribution Pre & Post-November 4th")
ax.legend(title="Sentiment")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "sentiment_pre_post_nov4.png",
    dpi=300,
    bbox_inches="tight"
)

