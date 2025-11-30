import pandas as pd 
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='csv input file')
    args = parser.parse_args()

    df = pd.read_csv(args.filepath)
    df.columns = df.columns.str.strip()  # Remove trailing spaces
    
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
    
    data = {}
    for category in categories:
        # Get articles for this category as a list
        category_articles = df[df['Category'] == category]['combined_text'].tolist()
        data[category] = category_articles
    
    # Create dataframe with columns of different lengths
    output_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    output_df.to_csv('articles_for_llm.csv', index=False)
    print("results saved")

if __name__ == "__main__":
    main()
