import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='CSV file from tf_idf.py (e.g., tf_idf_results_1grams.csv)')
    args = parser.parse_args()

    # Read the CSV
    df = pd.read_csv(args.filepath)
    
    # Pivot to create matrix: rows=words, columns=categories, values=scores
    heatmap_df = df.pivot(index='Word', columns='Category', values='tf-idf Score')
    
    # Fill missing values with 0 (word not in that category's top 10)
    heatmap_df = heatmap_df.fillna(0)
    
    # Sort by total score across all categories (most common words at top)
    heatmap_df['total'] = heatmap_df.sum(axis=1)
    heatmap_df = heatmap_df.sort_values('total', ascending=False).drop('total', axis=1)
    
    # : Keep only words that appear in 2+ categories
    heatmap_df['category_count'] = (heatmap_df > 0).sum(axis=1)
    heatmap_df = heatmap_df[heatmap_df['category_count'] >= 2]
    heatmap_df = heatmap_df.drop('category_count', axis=1)

    heatmap_df.index = heatmap_df.index.str.title()
    heatmap_df.index = heatmap_df.index.str.replace('_', ' ')

    heatmap_df = heatmap_df.sort_index()
    heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

    
    print(f"Showing {len(heatmap_df)} words that appear in 2+ categories")
    
    # Create heatmap with adjusted size for fewer rows
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(heatmap_df, cmap="YlOrBr", annot=False, 
                cbar_kws={'label': 'TF-IDF Score',}, 
                linewidths=0.0, linecolor='orange')

    # Get the colorbar from the heatmap
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)   
    cbar.ax.yaxis.label.set_size(10)   
    
    plt.title('Terms Appearing Across Multiple Categories', 
              fontsize=14, pad=10)
    plt.xlabel('Category', fontsize=15)
    plt.ylabel('Term', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11, rotation=0, ha='right')
    plt.tight_layout()
    plt.savefig('tfidf_cross_category_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved to tfidf_cross_category_heatmap.png")
    plt.show()

if __name__ == "__main__":
    main()

#python heatmap.py tf_idf_results_1grams.csv
