import pandas as pd
import requests
import cv2
import os
import re
import ast
from collections import Counter

import spacy
nlp = spacy.load("en_core_web_sm")

# Retrieve video thumbnail
def get_highest_quality_thumbnail(video_id):
    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
    response = requests.get(thumbnail_url)
    
    if response.status_code == 200:
        return thumbnail_url
    else:
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" # Fall back to hqdefault
    
def save_thumbnail(video_id):
    # Download & process the video thumbnail for analysis
    thumbnail_path = f"data/thumbnails/{video_id}.jpg"
    thumbnail_url = get_highest_quality_thumbnail(video_id)
    
    # Make sure the thumbnails directory exists
    os.makedirs(os.path.dirname(thumbnail_path), exist_ok = True)
    
    if not os.path.exists(thumbnail_path):
        img_data = requests.get(thumbnail_url).content
        with open(thumbnail_path, "wb") as handler:
            handler.write(img_data)
            
        # If we have hqdefault, crop off the black bars
        if "hqdefault" in thumbnail_url:
            thumbnail = cv2.imread(thumbnail_path)
            cropped_image = thumbnail[45:315, 0:480] # Crop to 480x270 (16:9 aspect ratio)
            resized_image = cv2.resize(cropped_image, (1280, 720), interpolation = cv2.INTER_AREA) # Resize to match with maxresdefault
            cv2.imwrite(thumbnail_path, resized_image)
            
    return thumbnail_path

# Determine video topic
def is_url_token(token):
    url_patterns = ["http", "www", ".com", ".net", ".org", ".io", ".gov", ".edu", ".ly"]
    return any(pattern in token for pattern in url_patterns)

def preprocess_text(text):
    entity_pass = nlp(text)
    entities = [entity.text for entity in entity_pass.ents]
    
    # Remove entities from this pass
    filtered_text = text
    for entity in entities:
        filtered_text = filtered_text.replace(entity, "")
        
    filtered_text = " ".join(filtered_text.split())
    
    nlp_text = nlp(filtered_text)
    tokens = [
        token.lemma_.lower()
        for token in nlp_text
        if not token.is_stop and not token.is_punct and len(token) > 2 and not is_url_token(token.text.lower())
    ]
    
    return (tokens, entities)

def process_tags(tag_list, entities):
    # Move longer entities to the front.
    # If there are any entities contained in others, these entities will be longer and therefore come first in the list.
    lowercase_entities = sorted([entity.lower() for entity in entities], key=len, reverse=True)
    
    refined_tags = []
    for tag in tag_list:
        entity_found = False
        
        for entity in lowercase_entities:
            if entity == tag:
                refined_tags.append(entity)
                entity_found = True
                break
            
            if entity in tag:
                # Split the entities from the rest of the string and extend results to the list
                segments = re.split(rf"({re.escape(entity)})", tag)
                for segment in segments:
                    segment = segment.strip()
                    
                    if segment and segment == entity:
                        refined_tags.append(segment)
                    else:
                        # Tokenize non-entity segments
                        nlp_text = nlp(segment)
                        tokens = [token.lemma_.lower() for token in nlp_text if not token.is_stop and not token.is_punct]
                        refined_tags.extend(tokens)
                
                entity_found = True
                break # Stop checking for other entities when a result is found
            
        if not entity_found:
            # Tokenize entire tag if no entity found
            nlp_text = nlp(tag)
            tokens = [token.lemma_.lower() for token in nlp_text if not token.is_stop and not token.is_punct and token.text.strip()]
            refined_tags.append(tag)
    
    refined_tags = [tag for tag in refined_tags if tag.strip()] # Remove empty elements & excessive whitespace
    return refined_tags

def determine_topic(topic_tokens, top_n=3):
    token_frequency = Counter(topic_tokens)
    sorted_tokens = [token for token, _ in token_frequency.most_common()]
    
    if sorted_tokens:
        return sorted_tokens[:top_n]
    else:
        return ["unknown"]

# Read the CSV File
def read_csv(csv):
    df = pd.read_csv(csv)
    processed_videos = []
    
    for _, row in df.iterrows():
        video = {}
        video_id = row["video_id"]
        title = row["title"]
        
        video["title"] = title
        video["thumbnail"] = save_thumbnail(video_id)
        video["vsr"] = row["vsr"]
                
        title_tokens, title_entities = preprocess_text(title)
        desc_tokens, desc_entities = preprocess_text(row["description"])
        entities = title_entities + desc_entities
        
        tags_list = ast.literal_eval(row["tags"]) # Make sure the tags are in list format (sometimes it's stored as a string in csv files)
        refined_tags = process_tags(tags_list, entities)
        topic_tokens = [token.lower() for token in title_tokens + desc_tokens + refined_tags]
        video["topic"] = determine_topic(topic_tokens)
        
        processed_videos.append(video)
            
    return processed_videos

if __name__ == "__main__":
    DIRECTORY = "data/raw"
    
    processed_videos = []
    for file in os.listdir(DIRECTORY):
        if file.endswith(".csv"):
            cvs_file = os.path.join(DIRECTORY, file)
            print(f"Processing {cvs_file}")
            processed_videos.extend(read_csv(cvs_file))
            
            break # Process one csv for debugging
            
    print(f"Finished processing {len(processed_videos)} videos")
    
    if processed_videos:
        processed_df = pd.DataFrame(processed_videos)
        processed_df.to_csv("data/processed_videos.csv", index=False)