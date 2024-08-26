from data_loader import load_data
from text_pipeline import text_pipeline

# Load the dataset
df = load_data('WIKIMEDIA_Text_Visualization/wiki_data.csv', n_rows=2000)

# Run the text processing pipeline
processed_text = text_pipeline.fit_transform(df['text_column'])
