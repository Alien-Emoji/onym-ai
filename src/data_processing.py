import nltk
from nltk.corpus import stopwords
import nltk.downloader
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

from gensim import corpora
from gensim.models import LdaModel

import pandas as pd
import requests
import cv2
import os

# Download NLTK files
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

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
            cv2.imwrite(thumbnail_path, cropped_image)
            
    return thumbnail_path

# Determine video topic
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation and word not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

def extract_topics():
    None

# Read the CSV File
def read_csv(csv):
    df = pd.read_csv(csv)
    processed_videos = []
    
    for _, row in df.iterrows():       
        video = {}
        video_id = row["video_id"]
        video_title = row["title"]
        
        video['title'] = video_title
        video['thumbnail'] = save_thumbnail(video_id)
        video['vsr'] = row['vsr']
        
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