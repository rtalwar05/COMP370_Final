import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("COMP370 Group Project Data - all_articles.csv")

    counts = df["Category "].value_counts()

    plt.figure(figsize=(7, 7))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=0)
    plt.title(f"Distribution of Article Categories")
    plt.show()


if __name__ == "__main__":
    main()