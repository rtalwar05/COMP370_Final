import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import argparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# List of words to exclude from analysis. 
# English Stop Words includes random words like and, the...
# The rest of the words are ones that I added after the initial analysis to clean up the results
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
    'zohran', 'mamdani', 'new', 'york', 'nyc', 'city', 'mayor', 'candidate', 'chars', 'words',
    'mayoral', 'said', 'post', 'elect', 'news', 'just', 'stay', 'past'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='csv input file')
    args = parser.parse_args()

    df = pd.read_csv(args.filepath)
    df.columns = df.columns.str.strip()
    #print(df.columns.tolist())

  # Combine all text components of the article (title, description and content snippet)
    df['combined_text'] = (
        df['title'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['content'].fillna('')
    )

    df = df[df['Category'].notna()]

    categories = [
        'Political Figure Opinions',
        'Political Identity',
        'Election + Campaign Results',
        'Popular Culture',
        'Affordability',
        'Immigration/ICE',
        'Personal Life',
        'Police & Crime'
    ]

    results =  {}
    print("tf-idf top 10 words by category\n")

    for category in categories:

        category_articles = df[df['Category'] == category]['combined_text'].tolist()

        if len(category_articles) == 0:
            print(f"uh oh no articles for {category}")
            continue

        # ngram_range can also be changed to (2, 2) to get bigrams
        vectorizer = TfidfVectorizer(
            max_features=100, 
            stop_words=CUSTOM_STOP_WORDS, 
            lowercase=True, 
            ngram_range=(1, 1)
        )

    
        tf_idf_matrix = vectorizer.fit_transform(category_articles)
        feature_names = vectorizer.get_feature_names_out()
        mean_tf_idf = np.asarray(tf_idf_matrix.mean(axis=0)).flatten()

        top_indices = mean_tf_idf.argsort()[-10:][::-1] #take last ten (highest), and reverse
        top_words = [(feature_names[i], mean_tf_idf[i]) for i in top_indices]


        results[category] = top_words

        print(f"\n{category} ({len(category_articles)} articles)")
        for i, (word, score) in enumerate(top_words, 1):
            print(f" {i:2d}. {word:20s} (tf-idf: {score:.4f})")

    # Save to csv
    output_data = []
    for category, words in results.items():
        for rank, (word, score) in enumerate(words, 1):
            output_data.append({
                'Category': category,
                'Rank': rank,
                'Word': word,
                'tf-idf Score': score
            })
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv('tf_idf_results_1grams.csv', index=False)
    print("Results saved to tf_idf_results.csv")

if __name__ == "__main__":
    main()


# Example call: tf_idf.py 'all_articles_annotated.csv'
