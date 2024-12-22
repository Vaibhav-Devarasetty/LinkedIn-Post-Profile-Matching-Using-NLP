# Importing all the libraries
import torch
import pandas as pd
import numpy as np
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel
import gensim
import gensim.downloader
from openai import OpenAI

import matplotlib.pyplot as plt

# Configure to the device you want your NN architecture to run in
device = torch.device('mps') 
#print(device)

# Setting up tokenizer and the model for generating embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.to(device)

def get_sentence_embedding(sentence, tokenizer, model):

    # Tokenize input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)

    # Pass input tokens through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the output of the [CLS] token (the first token)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding

# Calculating Cosine Similarities between sentences in bert
def cosine_similarity_bert(sentence1, sentence2, tokenizer, model):
    # Get embeddings for the sentences
    embedding1 = get_sentence_embedding(sentence1, tokenizer, model)
    embedding2 = get_sentence_embedding(sentence2, tokenizer, model)

    # Calculate cosine similarity
    cosine_similarity_bert_value = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1)
    return cosine_similarity_bert_value.item()

## Instead of running this, run the save embedding file below.
client = OpenAI(api_key='')

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    # if len(text) >= 8100:
    #     print("Huge")
    #     text = text[0:8100]
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def get_embedding_wrapper(topic_name):
    return get_embedding(topic_name, model='text-embedding-3-small')

#similarities = cosine_similarity([job_embedding], company_embeddings)[0]

def sentence_embedding(sentence, model):
    """
    Generate sentence embedding by averaging the word embeddings.
    """
    word_embeddings = []
    for word in sentence.split():
        if word in model:
            word_embeddings.append(model[word])
    if len(word_embeddings) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(word_embeddings, axis=0)

def cosine_similarity_model(sentence1, sentence2, model):
    """
    Calculate cosine similarity between two sentences using GloVe embeddings.
    """
    embedding1 = sentence_embedding(sentence1, model)
    embedding2 = sentence_embedding(sentence2, model)
    if np.all(embedding1 == 0) or np.all(embedding2 == 0):
        return 0  # Return 0 similarity if any of the sentences has no valid embeddings
    else:
        return cosine_similarity([embedding1], [embedding2])[0][0]

### The Below Codes are for checking out the cosine similarities for individual sentences
### These contain 4 different models which are OpenAI Model, Bert Representation, GloVe, Word2Vec
# Example sentences
#sentence1 = "The quick brown fox jumps over the lazy dog"
#sentence2 = "the slow black cat dives under the active tiger"

glove_vectors = gensim.downloader.load('glove-twitter-200')
word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')

# Calculate cosine similarity using all models
#similarity_glove = cosine_similarity_model(sentence1, sentence2, glove_vectors)
#similarity_word2vec = cosine_similarity_model(sentence1, sentence2, word2vec_vectors)
#similarity_openai = cosine_similarity([get_embedding(sentence1, model="text-embedding-3-small")], [get_embedding(sentence2, model="text-embedding-3-small")])[0][0]
#similarity_bert = cosine_similarity_bert(sentence1, sentence2, tokenizer, model)

#print("GloVe Similarity: ", similarity_glove)
#print("Word2Vec Similarity: ", similarity_word2vec)
#print("OpenAI Embedding Similarity: ", similarity_openai)
#print("Bert Embedding Similarity: ", similarity_bert)

def parse_string_to_list_of_dicts(string_data):
    # Safely evaluate the string as a Python literal
    parsed_data = ast.literal_eval(string_data)
    return parsed_data

### These are for getting embedding spaces of OpenAI for reducing the number of API drastically
### We dont have to do API call everytime for finding embeddings, just doing it beforehand once 
### It reduces the API calls number by the number of individual linkedin posts
key_phrases = [
    "AI-driven Code Generation",
    "Automated Code Review",
    "AI-based Bug Fixing",
    "Natural Language to Code Translation",
    "Dynamic Software Customization",
    "AI-enhanced Test Case Generation",
    "Predictive Development Analytics",
    "AI-optimized Refactoring",
    "Generative Models for UI Design",
    "AI-driven Development Pipelines"
]
key_phrases_embeddings = []
for key_phrase in key_phrases:
    key_phrases_embeddings.append(get_embedding(key_phrase, model="text-embedding-3-small"))

data = pd.read_csv("profile_post_similar_2.csv")

def parse_string_to_list_of_dicts(string_data):
    # Safely evaluate the string as a Python literal
    parsed_data = ast.literal_eval(string_data)
    return parsed_data

# empty_indices_list = []
# for i in range(len(data)):
#     if len(parse_string_to_list_of_dicts(data["Posts_data"][i])[0]) != 0:
#         for j in range(len(parse_string_to_list_of_dicts(data["Posts_data"][i])[0])):
#             sentence1 = parse_string_to_list_of_dicts(data["Posts_data"][i])[0][j]["Text"]
#             if(len(sentence1) == 0):
#                 indices = {}
#                 indices["user"] = i
#                 indices["post_number"] = j
#                 #print(f"User : {i}, Post Number : {j}")
#                 empty_indices_list.append(indices)
# empty_indices_list

sample_list = []
for i in range(len(data)):
    sample_list.append(None)
data["Glove_Similarities"] = sample_list
data["Word2Vec_Similarities"] = sample_list
data["Bert_Similarities"] = sample_list
data["OpenAI_Similarities"] = sample_list

for i in range(len(data)):
    if len(parse_string_to_list_of_dicts(data["Posts_data"][i])[0]) != 0:
        #print(f"i : {i}")
        similarities_bert = []
        similarities_openai = []
        similarities_glove = []
        similarities_word2vec = []
        for j in range(len(parse_string_to_list_of_dicts(data["Posts_data"][i])[0])):
            #print(f"j : {j}")
            sentence1 = parse_string_to_list_of_dicts(data["Posts_data"][i])[0][j]["Text"]
            if(len(sentence1) != 0):
                openai_1 = [get_embedding(sentence1, model="text-embedding-3-small")]
                similarity_bert = {}
                similarity_glove = {}
                similarity_word2vec = {}
                similarity_openai = {}

                for k in range(len(key_phrases)):
                    similarity_word2vec[key_phrases[k]] = cosine_similarity_model(sentence1, key_phrases[k], word2vec_vectors)
                    similarity_glove[key_phrases[k]] = cosine_similarity_model(sentence1, key_phrases[k], glove_vectors)
                    similarity_bert[key_phrases[k]] = cosine_similarity_bert(sentence1, key_phrases[k], tokenizer, model)
                    similarity_openai[key_phrases[k]] = cosine_similarity(openai_1, [key_phrases_embeddings[k]])[0][0]
                    
                similarities_bert.append(similarity_bert)
                similarities_glove.append(similarity_glove)
                similarities_word2vec.append(similarity_word2vec)
                similarities_openai.append(similarity_openai)

        # Add new rows to the DataFrame
        data["Bert_Similarities"][i] = similarities_bert
        data["Glove_Similarities"][i] = similarities_glove
        data["Word2Vec_Similarities"][i] = similarities_word2vec
        data["OpenAI_Similarities"][i] = similarities_openai

def evaluation_datapoint(data, column, row, post_number):
    datapoint = ast.literal_eval(data[column][row])[post_number]
    # Sort the dictionary items by value in descending order
    #sorted_data = sorted(datapoint.items(), key=lambda x: x[1], reverse=True)

    #print("Descending order of the dictionary:")
    # for key, value in sorted_data:
    #     print(f"{key}: {value}")

    # Find the key with the highest value
    max_key = max(datapoint, key=datapoint.get)
    max_value = datapoint[max_key]
    return max_key, max_value

similarity_keys = list(data.keys())[-4:]

# best_match, best_match_similarity = evaluation_datapoint(data, similarity_keys[3], 2, 3)
# best_match_similarity

# len(parse_string_to_list_of_dicts(data[similarity_keys[3]][i]))

sample_list = []
for i in range(len(data)):
    sample_list.append(None)
data["OpenAI_Best_Match"] = sample_list
data["OpenAI_Best_Match_Similarity"] = sample_list

for i in range(len(data)):
    if len(parse_string_to_list_of_dicts(data["Posts_data"][i])[0]) != 0:
        best_matches_of_user = []
        best_matches_similarity_of_user = []

        for post_number in range(len(parse_string_to_list_of_dicts(data["Posts_data"][i])[0])):
            #print(post_number)
            best_match, best_match_similarity = evaluation_datapoint(data, similarity_keys[3], i, post_number)
            best_matches_of_user.append(best_match)
            best_matches_similarity_of_user.append(best_match_similarity)

        data["OpenAI_Best_Match"][i] = best_matches_of_user
        data["OpenAI_Best_Match_Similarity"][i] = best_matches_similarity_of_user

def post_text(data, user, post_number):
    return (parse_string_to_list_of_dicts(data["Posts_data"][user])[0][post_number]["Text"])

cols = list(data.keys())
cols[1] = 'Post_text'

rows = []
for i in range(len(data)):
    if len(parse_string_to_list_of_dicts(data["Posts_data"][i])[0]) != 0:
        k = 0
        for j in range(len(parse_string_to_list_of_dicts(data["Posts_data"][i])[0])):
            text = parse_string_to_list_of_dicts(data["Posts_data"][i])[0][j]["Text"]

            if(len(text) != 0):
                row = { cols[0] : data[cols[0]][i],
                    cols[1] : text,
                    cols[2] : ast.literal_eval(data[cols[2]][i])[k],
                    cols[3] : ast.literal_eval(data[cols[3]][i])[k],
                    cols[4] : ast.literal_eval(data[cols[4]][i])[k],
                    cols[5] : ast.literal_eval(data[cols[5]][i])[k]
                }
                k += 1
                rows.append(row)

data = pd.DataFrame(rows)

cols = list(data.keys())

sample_list = []
for i in range(len(data)):
    sample_list.append(None)
data["Bert_Best_Match"] = sample_list
data["Bert_Best_Match_Similarity"] = sample_list

for i in range(len(data)):
    datapoint = ast.literal_eval(data[cols[4]][i])

    max_key = max(datapoint, key=datapoint.get)
    data["Bert_Best_Match"][i] = max_key
    data["Bert_Best_Match_Similarity"][i] = datapoint[max_key]

thresholds = [0.2, 0.225, 0.25, 0.275, 0.3, 0.325]

for threshold in thresholds:
    sample_list = []
    for i in range(len(data)):
        sample_list.append(None)
    attribute_name = f"Threshold_OpenAI : {threshold}"
    data[attribute_name] = sample_list
    for i in range(len(data)):
        if data["OpenAI_Best_Match_Similarity"][i] >= threshold:
            data[attribute_name][i] = True
        else:
            data[attribute_name][i] = False

#### The below is used for visualizing and analyzing different threshold values for OpenAI embeddings
#### It will help you determine which threshold
# # Define colors for True and False
# colors = ['#1f77b4', '#ff7f0e']  # Blue and orange colors

# # Threshold columns
# threshold_columns = ['Threshold_OpenAI : 0.2', 'Threshold_OpenAI : 0.225', 'Threshold_OpenAI : 0.25',
#                      'Threshold_OpenAI : 0.275', 'Threshold_OpenAI : 0.3', 'Threshold_OpenAI : 0.325']

# # Loop through each threshold column
# for column in threshold_columns:
#     # Calculate value counts
#     value_counts = data[column].value_counts()

#     # Plot pie chart with custom colors
#     plt.figure(figsize=(8, 6))
#     plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
#     plt.title(f'Distribution of {column}')
#     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     plt.show()

#data

### For Finding The OpenAI Best Match Counts which abstract stats and Visualization
# # 1. Value Counts
# value_counts = data['OpenAI_Best_Match'].value_counts()
# print("Value Counts:")
# print(value_counts)
# print()

# # 2. Descriptive Statistics
# statistics = data['OpenAI_Best_Match'].describe()
# print("Descriptive Statistics:")
# print(statistics)
# print()

# # 3. Plotting (Histogram)
# plt.figure(figsize=(8, 6))
# data['OpenAI_Best_Match'].value_counts().plot(kind='bar', color='skyblue')
# plt.title('Distribution of OpenAI Best Match')
# plt.xlabel('OpenAI Best Match')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.show()

# Assuming 'data' is your DataFrame

#### Grouping up OpenAI Best Match with similarities for detailed stats
# # Group by 'OpenAI_Best_Match' and calculate mean similarity for each group
# mean_similarity = data.groupby('OpenAI_Best_Match')['OpenAI_Best_Match_Similarity'].mean()

# # Print the mean similarity for each 'OpenAI_Best_Match' value
# print("Mean similarity for each OpenAI_Best_Match value:")
# print(mean_similarity)
# print()

# # Alternatively, you can also calculate other summary statistics like median, min, max, etc.
# median_similarity = data.groupby('OpenAI_Best_Match')['OpenAI_Best_Match_Similarity'].median()
# min_similarity = data.groupby('OpenAI_Best_Match')['OpenAI_Best_Match_Similarity'].min()
# max_similarity = data.groupby('OpenAI_Best_Match')['OpenAI_Best_Match_Similarity'].max()

# # Print other summary statistics
# print("Median similarity for each OpenAI_Best_Match value:")
# print(median_similarity)
# print()

# print("Minimum similarity for each OpenAI_Best_Match value:")
# print(min_similarity)
# print()

# print("Maximum similarity for each OpenAI_Best_Match value:")
# print(max_similarity)
# print()

# Assuming 'data' is your DataFrame

# # Group by 'OpenAI_Best_Match' and find the row with the maximum similarity for each group
# max_similarity_rows = data.loc[data.groupby('OpenAI_Best_Match')['OpenAI_Best_Match_Similarity'].idxmax()]

# # Iterate over each row to print the maximum similarity for each 'OpenAI_Best_Match' value with its corresponding 'Post_text'
# for index, row in max_similarity_rows.iterrows():
#     print("OpenAI_Best_Match:", row['OpenAI_Best_Match'])
#     print("Post_text:")
#     print(row['Post_text'])
#     print()

data_325 = data[data["Threshold : 0.325"] == True].reset_index(drop=True)
### for finding out the same for the highest threshold qualifying datapoints
# # Assuming 'data' is your DataFrame containing the 'OpenAI_Best_Match' column

# # 1. Value Counts
# value_counts = data_325['OpenAI_Best_Match'].value_counts()
# print("Value Counts:")
# print(value_counts)
# print()

# # 2. Descriptive Statistics
# statistics = data_325['OpenAI_Best_Match'].describe()
# print("Descriptive Statistics:")
# print(statistics)
# print()

# # 3. Plotting (Histogram)
# plt.figure(figsize=(8, 6))
# data_325['OpenAI_Best_Match'].value_counts().plot(kind='bar', color='skyblue')
# plt.title('Distribution of OpenAI Best Match')
# plt.xlabel('OpenAI Best Match')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.show()

#### Human Based Qualitative Analysis
# k = 0
# print(data_325['OpenAI_Best_Match'][k])
# print(data_325["Post_text"][k])
# print(data_325["OpenAI_Best_Match_Similarity"][k])

### For clearing the words which are very biased towards a specific key phrase
def remove_words(text):
    # Define the words to remove
    words_to_remove = ["microsoft", "azure", "saas", "cloud", "architecture"]

    # Create a regular expression pattern to match any form of the words
    pattern = r'\b(?:' + '|'.join(map(re.escape, words_to_remove)) + r')\b'

    # Replace occurrences of the words with an empty string
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return cleaned_text

#### For finding the matching keyphrase for a specific text using OpenAI embeddings
def similarity_with_key_phrases(text1, key_phrases_embeddings):
    openai_1 = [get_embedding(text1, model="text-embedding-3-small")]
    similarities_openai = {}

    for k in range(len(key_phrases)):
        similarities_openai[key_phrases[k]] = cosine_similarity(openai_1, [key_phrases_embeddings[k]])[0][0]

    max_key = max(similarities_openai, key=similarities_openai.get)
    max_value = similarities_openai[max_key]

    print(f"Key : {max_key}")
    print(text1)
    print(f"Score : {max_value}")

thresholds_bert = [0.75, 0.775, 0.8, 0.825, 0.85]

for threshold in thresholds_bert:
    sample_list = []
    for i in range(len(data)):
        sample_list.append(None)
    attribute_name = f"Threshold_Bert : {threshold}"
    data[attribute_name] = sample_list
    for i in range(len(data)):
        if data["Bert_Best_Match_Similarity"][i] >= threshold:
            data[attribute_name][i] = True
        else:
            data[attribute_name][i] = False

### Qualitative Analysis of the highest threshold for bert embeddings
data_85 = data[data["Threshold_Bert : 0.85"] == True].reset_index(drop=True)

### Human Based Qualitative Analysis of highest threshold in bert embeddings
# k = 2
# print(data_85['Bert_Best_Match'][k])
# print(data_85["Post_text"][k])
# print(data_85["Bert_Best_Match_Similarity"][k])

data.to_csv("Final_Similarity_Scores.csv", index=False)
