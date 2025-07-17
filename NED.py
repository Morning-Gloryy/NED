from underthesea import ner
from wikipedia import *
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
from tqdm import tqdm
from glob import glob
from joblib import Memory
import pandas as pd 
import numpy as np
import torch
import re

model_name = 'vinai/phobert-base-v2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
phobert_model = AutoModel.from_pretrained(model_name)

embedding_cache1 = Memory("cache/embeddingsPhoBert", verbose=0)
    
def load_csv():
    df = pd.read_csv('content/dataset.csv', encoding='utf-8', sep=";")
    df = df.replace('\n', '', regex=True)
    return df

def load_abbreviation(path='lib/abbreviation.xlsx'):
    ab = pd.read_excel(path)
    abbreviations = dict(zip(ab['Abrreviation'], ab['Full']))
    return abbreviations 

def load_dictionary(path='lib/Dictionary_specialized_word.xlsx'):
    df = pd.read_excel(path)
    return df
   
@embedding_cache1.cache
def encode_sentences_PhoBert(sentences, batch_size=32):
    vectors = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = phobert_model(**inputs)
            batch_vecs = outputs.last_hidden_state[:, 0, :].numpy()
            vectors.append(batch_vecs)
    return np.vstack(vectors)

def cosine_distance(vector1, vector2):
    # Compute cosine similarity
    similarity = cosine_similarity([vector1], [vector2])
    return 1 - similarity[0][0]

def applyNED():
    # load files
    abbreviations = load_abbreviation()
    df = load_csv()
    dictionary = load_dictionary()
    
    dict_entities = dictionary["Entity"].tolist()
    
    cleaned_entities = []
    # Remove space and other types value
    for s in dict_entities:
        if isinstance(s, str):
            trimmed = s.strip()
            if trimmed:
                cleaned_entities.append(trimmed)
    dict_entities = cleaned_entities
    
    dict_texts = dictionary["Text"].tolist()

    # Encode the dictionary
    print("Encoding Dictionary with PhoBert")
    dict_vecs = encode_sentences_PhoBert(dict_entities)
    dict_vecs = np.array(dict_vecs)
    lower_dict = [d.lower() for d in dict_entities]

    results = []
    augmented_texts = []
    seen = set()

    for text in tqdm(df['text'], desc="Applying NED"):
        entities = ner(text)
        
        # Extract Noun entities
        noun_entities = [e[0] for e in entities if e[1] in ["N", "Np", "Nb"]]
        # Remove number
        noun_entities = [ent for ent in noun_entities if not re.search(r"\d", ent)]
        # Lowercase entity
        noun_entities = [ent.lower() for ent in noun_entities]
        # Check abbreviation
        noun_entities = [abbreviations.get(ent.strip(), ent.strip()) for ent in noun_entities]
        
        # Remove if its substring
        filtered_entities = []
        for ent in noun_entities:
            if not any((ent != other and ent in other) for other in noun_entities):
                filtered_entities.append(ent)
        noun_entities = filtered_entities
        
        # Skip if there are no entity
        if not noun_entities:
            augmented_texts.append(text)
            continue

        # Encode with PhoBert
        entity_vecs = encode_sentences_PhoBert(noun_entities)
        entity_vecs = np.array(entity_vecs)
        
        # Ensure shape
        if entity_vecs.ndim == 1:
            entity_vecs = entity_vecs.reshape(1, -1)
        if dict_vecs.ndim == 1:
            dict_vecs = dict_vecs.reshape(1, -1)
           
        # Compare with dictionary 
        similarities = cosine_similarity(entity_vecs, dict_vecs)  # batch similarity
        augmented_text = text


        for index, original_entity in enumerate(noun_entities):
            if original_entity in seen:
                continue
            seen.add(original_entity)

            # Get most similar dictionary entry
            sim_row = similarities[index]
            best_index = np.argmax(sim_row)
            best_score = sim_row[best_index]
            best_dist = 1 - best_score

            match_type = None
            matched_entity, matched_text = None, None
            
            # Check for distance between two entities
            if best_dist < 0.1:
                matched_entity = dict_entities[best_index]
                matched_text = dict_texts[best_index]
                match_type = "cosine"
            # Fall back to text matching
            elif original_entity.lower() in lower_dict:
                fallback_idx = lower_dict.index(original_entity.lower())
                matched_entity = dict_entities[fallback_idx]
                matched_text = dict_texts[fallback_idx]
                match_type = "exact"

            # Add Annotate into the text string
            if matched_text:
                annotated = f"{original_entity} ({matched_text})"
                augmented_text = re.sub(rf"\b{re.escape(original_entity)}\b", annotated, augmented_text, count=1)

                # Report NED processing
                results.append({
                    "OriginalEntity": original_entity,
                    "MatchedEntity": matched_entity,
                    "DictionaryText": matched_text,
                    "Distance": best_dist if match_type == "cosine" else None,
                    "OriginalText": text,
                    "MatchType": match_type
                })
        
        augmented_texts.append(augmented_text)
        
    pd.DataFrame(results).sort_values(by='OriginalEntity').to_csv("result/ned_results.csv", index=False, encoding="utf-8-sig")
    df['text'] = augmented_texts
    df.to_csv("content/datasetNED.csv", index=False, encoding="utf-8-sig", sep=";")   
    
def main():
    applyNED()
    
if __name__ == "__main__":
    main()




