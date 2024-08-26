import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import easyocr
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
import requests
import math
import cv2
import os

# Initialize stuff
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()
reader = easyocr.Reader(['en'], gpu = True)

# Clean the Data
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

# Extract title features: Length, Keywords, Sentiment
def get_title_NLP_features(title):
    tokens = nltk.word_tokenize(title)
    sentiment = sid.polarity_scores(title)
    
    return (tokens, sentiment)

# Extract thumbnail features: Color Analysis (brightness, saturation, contrast), Text Presence, Object Detection
def get_thumbnail_attributes(thumbnail_path):
    thumbnail = Image.open(thumbnail_path)
    thumbnail_np = np.array(thumbnail)
    
    # Get perceived brightness value (from https://stackoverflow.com/a/3498247/26443787)
    stat = ImageStat.Stat(thumbnail)
    r,g,b = stat.mean[:3]
    perceived_brightness = math.sqrt(
        0.241 * (r ** 2) +
        0.691 * (g ** 2) +
        0.068 * (b ** 2)
    )
    
    # Get the thumbnail's mean saturation value
    hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean()
    
    # Get the thumbnail's contrast value
    grayscale = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
    contrast = grayscale.std()
    
    return (perceived_brightness, saturation, contrast)

def detect_thumbnail_text(thumbnail_path):
    thumbnail = cv2.imread(thumbnail_path)
    results = reader.readtext(thumbnail)
    return " ".join([text for _, text, _ in results])

def detect_thumbnail_objects(thumbnail_path):
    None

# Title & Thumbnail synergy score:

# Normalize data

# Read the CSV File
def read_csv(csv):
    df = pd.read_csv(csv)
    processed_videos = []
    
    for index, row in df.iterrows():       
        video = {}
        video_id = row["video_id"]
        video_title = row["title"]
        
        thumbnail_path = save_thumbnail(video_id)
        brightness, saturation, contrast = get_thumbnail_attributes(thumbnail_path)
        tokens, sentiment = get_title_NLP_features(video_title)
        thumbnail_text = detect_thumbnail_text(thumbnail_path)
        
        video['title'] = video_title # For debugging, remove later
        video['vsr'] = row['vsr']
        video['title_length'] = len(tokens)
        video['title_neg'] = sentiment['neg']
        video['title_neu'] = sentiment['neu']
        video['title_pos'] = sentiment['pos']
        video['title_abs_compound'] = abs(sentiment['compound']) # We use the absolute sentiment compound to gauge how much emotion is in the title
        video['thumbnail_brightnses'] = brightness
        video['thumbnail_saturation'] = saturation
        video['thumbnail_contrast'] = contrast
        
        if thumbnail_text:
            print(f"({video_id}) Text detected in the thumbnail: {thumbnail_text}") # For debugging
        
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