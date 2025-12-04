import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import argparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# List of words to exclude from analysis
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
    'zohran', 'mamdani', 'new', 'york', 'nyc', 'city', 'mayor', 'candidate', 'chars', 'words', 'daily', 'video', 'interview', 'mayoral', 'gov', 'tuesday', 'friday',
    'mayoral', 'said', 'post', 'elect', 'elected', 'news', 'just', 'stay', 'past', 'leading', 'year', 'old', 'li', 'lead', 'wednesday', '2025'
]

# First round of concatenating -- replace names that appear as either first or last with just last
PHRASE_MAP_1 = {
    'donald': 'trump',
    'rama': 'duwaji',
}

# Second round of concatenating -- replace multi-word terms with single term for tf-idf
PHRASE_MAP = {
    "trump": "donald_trump",
    "president": "donald_trump",
    "donald trump": "donald_trump",
    "jessica tisch": "jessica_tisch",
    "curtis sliwa": "curtis_sliwa",
    'eric adams': 'eric_adams',
    'andrew cuomo': 'andrew_cuomo',
    'rama duwaji': 'rama_duwaji',
    'rama': 'rama_duwaji',
    'duwaji': 'rama_duwaji',
    'mira nair': 'mira_nair',
    'white house': 'white_house',
    'oval office': 'oval_office',
    'new yorkers': 'new_yorkers',
    'hakeem jeffries': 'hakeem_jeffries',
    'cops': 'cop'

}

def preprocess_text(text, phrase_map):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Replace multi-word phrases with single-token names
    for phrase, token in phrase_map.items():
        text = text.replace(phrase, token)

    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='csv input file')
    args = parser.parse_args()

    df = pd.read_csv(args.filepath)
    df.columns = df.columns.str.strip()
    #print(df.columns.tolist())

    df['combined_text'] = (
        df['title'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['content'].fillna('')
    )

    df = df[df['Category'].notna()]

    categories = [
        'Political Figure Opinions',
        'Political Identity',
        'Election + Campaign Results ',
        'Popular Culture',
        'Affordability',
        'Immigration/ICE',
        'Personal Life',
        'Police & Crime'
    ]

    results =  {}
    print("tf-idf top 10 words by category\n")

    for category in categories:

        raw_articles = df[df['Category'] == category]['combined_text'].tolist()
        first_category_articles = [preprocess_text(t, PHRASE_MAP_1) for t in raw_articles]
        category_articles = [preprocess_text(t, PHRASE_MAP) for t in first_category_articles]


        if len(category_articles) == 0:
            print(f"uh oh no articles for {category}")
            continue

        vectorizer = TfidfVectorizer(
            max_features=100, 
            stop_words=CUSTOM_STOP_WORDS, 
            lowercase=True, 
            ngram_range=(1, 1),
            norm=None
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

    #python tf_idf.py all_articles_annotated.csv
