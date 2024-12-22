# LinkedIn Posts Matching Project

## Abstract
This project aims to develop a system for matching LinkedIn posts with specific profiles. Leveraging advanced natural language processing techniques and embeddings, it processes LinkedIn profiles and posts data to find relevant matches. This project explores various embedding models, search strategies, and visualization techniques to achieve optimal results.

## Project Description
The core objective of this project is to analyze and match LinkedIn posts with profiles based on specific criteria. It utilizes a variety of embeddings (OpenAI, BERT, GloVe, Word2Vec) and search strategies (semantic and hybrid searches) to measure the similarity between posts and key phrases.

## Requirements
- Python 3.8+
- pandas
- numpy
- ast
- re
- scikit-learn
- transformers
- gensim
- openai
- matplotlib
- torch

To install the necessary packages, run:
```bash
pip install pandas numpy ast re scikit-learn transformers gensim openai matplotlib torch
```

## Usage
1. **linkedin_posts_matching.py**: This script is responsible for matching LinkedIn posts with profiles.
2. **lpm_function.py**: This script contains the functions used by `linkedin_posts_matching.py`.
3. **linkedin_person_profile_posts_data.csv**: This CSV file contains the LinkedIn profiles and posts data.

## Code Structure
- `linkedin_posts_matching.py`: Main script to run the LinkedIn posts matching.
- `lpm_function.py`: Contains the core functions used in the main script.
- `linkedin_person_profile_posts_data.csv`: Input data file containing LinkedIn profiles and their posts.

## File Descriptions
### linkedin_posts_matching.py
This script processes the LinkedIn profile data and matches it with relevant posts. It reads input data from `linkedin_person_profile_posts_data.csv` and uses functions defined in `lpm_function.py`.

### lpm_function.py
Contains core functions used for embedding generation, similarity calculations, and other utilities required by `linkedin_posts_matching.py`.

### linkedin_person_profile_posts_data.csv
This CSV file includes columns for profile information and LinkedIn posts. Each row represents a LinkedIn post associated with a profile.

## Input and Output
- **Input**: The input data is read from `linkedin_person_profile_posts_data.csv`.
- **Output**: The output is generated in the form of matched posts printed on the console or saved to an output file.

## Pipeline Implementation
1. **Preprocessing**: Scripts developed to enhance data processing efficiency.
2. **Embedding Models**: Utilized embeddings such as OpenAI, BERT, GloVe, and Word2Vec to measure similarities between posts and key phrases.
3. **Search Strategies**: Explored various search strategies, including semantic and hybrid searches.
4. **Statistical Analysis**: Conducted statistical analysis to determine optimal thresholds tailored to different company contexts.
5. **Visualization**: Implemented code for detailed visualization of both data and results, ensuring clarity and precision.

## How to Run the Project
1. Ensure all requirements are installed.
2. Place `linkedin_person_profile_posts_data.csv` in the same directory as the scripts.
3. Run `linkedin_posts_matching.py`:
   ```bash
   python linkedin_posts_matching.py
   ```
4. The script will process the data and output the matched posts.

## Detailed Guide

### linkedin_posts_matching.py
This script includes the following key steps:
1. **Import Libraries**: Imports necessary libraries for data processing, embedding generation, and similarity calculations.
2. **Load Data**: Reads the CSV file containing LinkedIn profiles and posts.
3. **Generate Embeddings**: Utilizes various embedding models to generate sentence embeddings.
4. **Calculate Similarity**: Measures cosine similarity between the embeddings of posts and profiles.
5. **Output Results**: Outputs the matched posts.

### lpm_function.py
Contains functions such as:
- `get_sentence_embedding`: Generates sentence embeddings using a specified model and tokenizer.
- `calculate_similarity`: Computes cosine similarity between embeddings.
- Other utility functions required for data processing and analysis.
