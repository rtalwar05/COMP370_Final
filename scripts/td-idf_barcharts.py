import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("COMP370 Group Project Data - tf_idf_results_final.csv")

    categories = df["Category"].unique()


    #Generate one bar chart for each category
    for cat in categories:
        subset = df[df["Category"] == cat].sort_values("tf-idf Score", ascending=True)

        plt.figure(figsize=(8, 5))
        plt.barh(subset["Word"], subset["tf-idf Score"], color="skyblue")
        for i, (score, word) in enumerate(zip(subset["tf-idf Score"], subset["Word"])):
            plt.text(score + 0.02, i, f"{score:.2f}", va='center')
        plt.title(f"TF-IDF Scores for {cat}")
        plt.rc('font', size=18)
        plt.xlabel("TF-IDF Score")
        plt.ylabel("Word")

        #standard x axis scale
        plt.xlim (0,4.5)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()