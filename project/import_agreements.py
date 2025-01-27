import os
import pandas as pd
from docx import Document
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
from ollama import chat
import nltk
from nltk.tokenize import sent_tokenize

#--------------------DANE WSADOWE--------------------
MY_QUESTION = 'Puzzle Games: Explain to me how should we calculate revenue share for Puzzle With Us game?'
YOUR_CONTEXT = 'You are a financial assistant. Answer the question using the provided context from legal agreements. Include the document reference in your response.'
ENCODER = 'all-mpnet-base-v2'
MODEL_SELECTED = 'smallthinker'
SENTENCES = 10


#--------------------ŹRÓDŁO DOCX--------------------
folder_path = 'C:\KSEYDA\REPOZYTORIA\LLM_KSeyda\docs'

agreements = []

# Iteracja przez wszystkie pliki w folderze
for file_name in os.listdir(folder_path):
    if file_name.endswith('.docx'):  # Filtruj pliki docx
        file_path = os.path.join(folder_path, file_name)

        # Odczyt pliku docx
        doc = Document(file_path)
        content = []
        for paragraph in doc.paragraphs:
            content.append(paragraph.text)

        # Zapisanie treści do listy
        agreements.append({
            'file_name': file_name,
            'content': "\n".join(content)
        })

# Konwersja listy na DataFrame
data = pd.DataFrame(agreements)

#Podział na zdania
sentences = []
for _, row in data.iterrows():
    doc_number = row["file_name"]
    for sentence in sent_tokenize(row["content"]):
        sentences.append({"file_name": doc_number, "sentence": sentence})

data = pd.DataFrame(sentences)
print(data.head())

print("\nQUESTION: ",MY_QUESTION)

#--------------------TWORZENIE WEKTORÓW--------------------
encoder = SentenceTransformer(ENCODER) # Model to create embeddings
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance


#--------------------DOCUMENT-LEVEL--------------------
# Check if the collection exists, and create it if it doesn't
if not qdrant.collection_exists(collection_name="publishing_agreements"):
    qdrant.create_collection(
        collection_name="publishing_agreements",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size defined by the model
            distance=models.Distance.COSINE  # Cosine distance
        )
    )

#Document-Level Embedding
document_points = [
    models.PointStruct(
        id=idx,
        vector=encoder.encode(row['sentence']).tolist(),  # Embed the entire document
        payload={"file_name": row["file_name"], "sentence": row["sentence"]}
    )
    for idx, row in data.iterrows()
]

qdrant.upload_points(
    collection_name="publishing_agreements",
    points=document_points
)

document_hits = qdrant.query_points(
    collection_name="publishing_agreements",
    query=encoder.encode(MY_QUESTION).tolist(),
    limit=1  # Get the top agreement
)

relevant_documents = [hit.payload for hit in document_hits.points]

relevant_file_name = [doc['file_name'] for doc in relevant_documents]
print("Document that will be a source of information: ",relevant_file_name)

filtered_sentences = [
    {"file_name": row['file_name'], "sentence": row['sentence']}
    for _, row in data.iterrows()
    if row['file_name'] in relevant_file_name
]


#--------------------SENTENCE-LEVEL--------------------
if not qdrant.collection_exists(collection_name="sentence_collection"):
    qdrant.create_collection(
        collection_name="sentence_collection",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size defined by the model
            distance=models.Distance.COSINE  # Cosine distance
        )
    )


# Sentence-Level Embeddings
sentence_points = [
    models.PointStruct(
        id=idx,
        vector=encoder.encode(sentence['sentence']).tolist(),
        payload=sentence
    )
    for idx, sentence in enumerate(filtered_sentences)
]

qdrant.upload_points(
    collection_name="sentence_collection",
    points=sentence_points
)


hits = qdrant.query_points(
    collection_name="sentence_collection",
    query=encoder.encode(MY_QUESTION).tolist(),  # Encode the query
    limit=SENTENCES  # Number of results to retrieve
)


search_results = [
    {"doc_number": point.payload['file_name'], "content": point.payload['sentence']}
    for point in hits.points
]

#--------------------POŁĄCZENIE Z LLM--------------------

available_models = ollama.list()
available_model_names = [model.model for model in available_models.models]

# Pull model if it is not available
if (MODEL_SELECTED not in available_model_names):
    print('Selected model pulled:', MODEL_SELECTED)
    ollama.pull(MODEL_SELECTED)


#Question to the model

search_results_str = "\n\n".join(
    [f"Document: {item['doc_number']}\nContent: {item['content']}" for item in search_results]
)

response = chat(
    model=MODEL_SELECTED,
    messages=[
        {'role': 'system', 'content': YOUR_CONTEXT},
        {'role': 'user', 'content': f"My question: {MY_QUESTION}"},
        {'role': 'assistant', 'content': f"Context:\n{search_results_str}"},
    ]
)
print(response)
print('Response:', response['message']['content'])