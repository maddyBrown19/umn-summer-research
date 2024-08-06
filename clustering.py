from openai import OpenAI
import numpy as np
import pandas as pd
import tiktoken
from ast import literal_eval
from sklearn.cluster import KMeans
import random
from main import query_GPT_without_system_role

client = OpenAI()
embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000
number_of_clusters = 10

"""IMPORTANT: You need to follow this link: https://drive.google.com/open?id=1t5tWWv5xkkU-YerJZabu-QMBajaqAnDB
to download the Reddit data on your computer. Make sure to name the file 'AskDocs.csv' so it matches the notation
below and the function works. AskDocs dataset GitHub cited here: https://github.com/fostiropoulos/AskDoc"""
def read_data():
    """Reads in Reddit AskDocs data and returns it in a DataFrame."""
    file = "AskDocs.csv"
    df = pd.read_csv(file)
    return df

def get_questions():
    """Extracts the questions from the Reddit data and returns them in a list."""
    df = read_data()
    questions = []
    for index, row in df.iterrows():
        question = row["Question"]
        questions.append(question)
    return questions

def get_questions_df(): 
    """Returns 1,000 Reddit questions that fit within the maximum token limit defined at the top of this file."""
    questions = get_questions()
    header = ["Question"]
    df = pd.DataFrame(questions)
    df.columns = header
    tokens = []
    for index, row in df.iterrows():
        encoding = tiktoken.get_encoding(embedding_encoding)
        tokens.append(len(encoding.encode(str(row["Question"]))))
    df["Tokens"] = tokens
    return df[df.Tokens <= max_tokens].head(1000)

def get_embedding(text):
    """Returns the embedding for a given string of text."""
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model = embedding_model).data[0].embedding

def make_embedding_csv():
    """For each question, generate an embedding for the question and store it in the DataFrame that contains the questions.
    Then, convert the DataFrame into a CSV file."""
    df = get_questions_df()
    embeddings = []
    for index, row in df.iterrows():
        embeddings.append(get_embedding(row["Question"]))
    df["Embedding"] = embeddings
    df.to_csv("embeddings.csv")

def get_matrix():
    """Returns a matrix of the question embeddings."""
    df["Embedding"] = df.Embedding.apply(literal_eval).apply(np.array)
    matrix = np.vstack(df.Embedding.values)
    return matrix

def get_clusters(): 
    """Sorts the Reddit questions into numerical clusters by similarity using k-means and stores cluster labels in the 
    DataFrame containing the questions and their embeddings."""
    kmeans = KMeans(n_clusters = number_of_clusters, init = "k-means++", random_state = 42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df["Cluster"] = labels
    df.groupby("Cluster").Question

def get_cluster_themes():
    """Print qualitative descriptions of the themes of each cluster by sampling medical questions from each cluster 
    and asking GPT-3.5 to determine what the samples in each cluster have in common.""" 
    questions_per_cluster_to_sample = 5
    random.seed(20)
    results = []
    for i in range(number_of_clusters):
        cluster_questions = []
        for index, row in df.iterrows():
            if row["Cluster"] == i:
                cluster_questions.append(row["Question"])
        results.append((i, cluster_questions))
    samples = []
    for result in results:
        cluster = result[0]
        questions = result[1]
        random_questions = []
        for i in range(questions_per_cluster_to_sample):
            random_questions.append(random.choice(questions))
        samples.append((cluster, random_questions))
    themes = []
    for sample in samples:
        user_prompt = "What do the following medical questions have in common? Questions: " + str(sample[1]) + "Theme: "
        themes.append(query_GPT_without_system_role(user_prompt))
    for theme in themes:
        print(theme + "\n")
    

"""Uncomment below to generate cluster themes."""
#make_embedding_csv()
#df = pd.read_csv("embeddings.csv")
#matrix = get_matrix()
#get_clusters()
#get_cluster_themes()
