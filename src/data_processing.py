# pylint: disable=E1101
# pylint: disable=C0303
# pylint: disable=C0116

import os
import re
import ast
from collections import Counter
import logging
import asyncio
import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError
import numpy as np
import pandas as pd
import cv2
import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
nlp = spacy.load("en_core_web_sm")

# Constants
REQUIRED_COLUMNS = ["vsr", "description", "title"]
CROP_TOP = 45
CROP_BOTTOM = 315
CROP_WIDTH = 480
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Retrieve video thumbnail
async def get_highest_quality_thumbnail(video_id: str, session: aiohttp.ClientSession, timeout: int = 5) -> str:
    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
    try:
        async with session.get(thumbnail_url, timeout=timeout) as response:
            if response.status == 200:
                return thumbnail_url
    except (aiohttp.ClientError, asyncio.TimeoutError):
        logging.warning("Request for maxresdefault thumbnail failed for %s. Falling back to hqdefault.", video_id)
        
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" # Fallback to hqdefault

@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(5), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
async def download_thumbnail(session: aiohttp.ClientSession, url: str, timeout: int = 5) -> bytes:
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                return await response.read()
            elif response.status in {500, 502, 503, 504}:
                raise aiohttp.ClientError(f"Server error: {response.status}")
            else:
                logging.warning("Failed to download %s with status code: %s", url, response.status)
                return None
    except asyncio.TimeoutError:
        logging.warning("Request to %s timed out.", url)
        raise
    except aiohttp.ClientError as e:
        logging.warning("Client error occured for %s: %s", url, e)
        raise
    
async def download_thumbnail_with_fallback(session: aiohttp.ClientSession, url: str, timeout: int = 5) -> bytes:
    try:
        result = await download_thumbnail(session, url, timeout)
        return result
    except RetryError:
        logging.error("Failed to download thumbnail: %s", url)
        return None

async def save_thumbnail(video_id: str, session: aiohttp.ClientSession, timeout: int = 5) -> str:
    # Download & process the video thumbnail for analysis
    thumbnail_path = f"data/thumbnails/{video_id}.jpg"
    
    if os.path.exists(thumbnail_path):
        logging.info("Thumbnail already exists for %s", video_id)
        return thumbnail_path
    
    thumbnail_url = await get_highest_quality_thumbnail(video_id, session)
    response = await download_thumbnail_with_fallback(session, thumbnail_url, timeout)
    
    if response is None:
        logging.warning("Failed to download or decode thumbnail for %s", video_id)
        return None
    
    img_data = np.frombuffer(response, np.uint8)
    thumbnail = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    
    if thumbnail is not None:
        # If we have hqdefault, crop off the black bars
        if "hqdefault" in thumbnail_url:
            cropped_image = thumbnail[CROP_TOP:CROP_BOTTOM, 0:CROP_WIDTH] # Crop to 480x270 (16:9 aspect ratio)
            resized_image = cv2.resize(cropped_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imwrite(thumbnail_path, resized_image)
        else:
            cv2.imwrite(thumbnail_path, thumbnail) # Directly save maxresdefault thumbnails
    else:
        logging.warning("Failed to decode thumbnail for %s", video_id)
        return None
        
    return thumbnail_path

# Determine video topic
def is_url_token(token: str) -> bool:
    url_patterns = ["http", "www", ".com", ".net", ".org", ".io", ".gov", ".edu", ".ly"]
    return any(pattern in token for pattern in url_patterns)

def extract_entities_and_tokens(doc) -> tuple[list[str], list[str]]:
    # Extract entities & store their start and end positions
    entities = [entity.text for entity in doc.ents]
    entity_tokens = {token.idx for entity in doc.ents for token in entity}
    
    # Exclude non-entity tokens
    filtered_text = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and len(token) > 2 and token.idx not in entity_tokens
    ]
    
    return filtered_text, entities

def tokenize_texts(texts: list[str]) -> list[tuple[list[str], list[str]]]:
    docs = nlp.pipe(texts)
    results = []
    
    for doc in docs:
        tokens, entities = extract_entities_and_tokens(doc)
        results.append((tokens, entities))
        
    return results

def tokenize_tags(tag_list: list[str], entities: list[str]) -> list[str]:
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
                        doc = nlp(segment)
                        tokens, _ = extract_entities_and_tokens(doc)
                        refined_tags.extend(tokens)
                
                entity_found = True
                break # Stop checking for other entities when a result is found
            
        if not entity_found:
            # Tokenize entire tag if no entity found
            doc = nlp(tag)
            tokens, _ = extract_entities_and_tokens(doc)
            refined_tags.append(tag)
    
    refined_tags = [tag for tag in refined_tags if tag.strip()] # Remove empty elements & excessive whitespace
    return refined_tags

def determine_topic(topic_tokens: list[str], top_n: int = 3) -> list[str]:
    token_frequency = Counter(topic_tokens)
    sorted_tokens = [token for token, _ in token_frequency.most_common()]
    
    return sorted_tokens[:top_n] if sorted_tokens else ["unknown"]

# Read the CSV File
async def read_csv(csv: str) -> list[dict]:
    try:
        df = pd.read_csv(csv)
    except pd.errors.EmptyDataError:
        logging.error("%s is empty.", csv)
        return []
    except FileNotFoundError:
        logging.error("Couldn't find file: %s", csv)
        return []
    except pd.errors.ParserError:
        logging.error("Couldn't parse %s", csv)
        return []
    except Exception as e:
        logging.error("Unexpected error while reading %s: %s", csv, e)
        return []
    
    # Handle missing columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        logging.error("%s is missing columns %s", csv, ", ".join(missing_columns))
        return []
    
    # Filter out rows with missing or empty required fields
    df = df.dropna(subset=["vsr", "description", "title"])
    df = df[df["title"].str.strip() != ""]
    df = df.reset_index(drop=True)
    
    # Group titles and descriptions for batch processing
    titles = df["title"].tolist()
    descriptions = df["description"].tolist()
    
    title_results = tokenize_texts(titles)
    desc_results = tokenize_texts(descriptions)
    
    # Download video thumbnails asynchronously
    os.makedirs(os.path.dirname("data/thumbnails"), exist_ok=True)
    async with aiohttp.ClientSession() as session:
        tasks = [
            save_thumbnail(row["video_id"], session)
            for _, row in df.iterrows()
        ]
        thumbnail_paths = await asyncio.gather(*tasks)
    
    processed_videos = []
    for idx, row in df.iterrows():
        if thumbnail_paths[idx] is None:
            logging.warning("Skipping video %s due to missing thumbnail", row["video_id"])
            continue

        video = {}
        title = row["title"]
        video["title"] = title
        video["thumbnail"] = thumbnail_paths[idx]
        video["vsr"] = row["vsr"]
        
        title_tokens, title_entities = title_results[idx]
        desc_tokens, desc_entities = desc_results[idx]
        entities = title_entities + desc_entities
        
        # Ensure the tags are formatted in a list
        try:
            tags_list = ast.literal_eval(row["tags"])
            if not isinstance(tags_list, list):
                raise ValueError(f"Tags for video {row['video_id']} are not a list.")
        except (ValueError, SyntaxError) as e:
            logging.error("Couldn't parse tags for video %s: %s", row["video_id"], e)
            tags_list = []
        
        tag_tokens = tokenize_tags(tags_list, entities)
        
        # Combine all tokens & determine the topic
        topic_tokens = [token.lower() for token in title_tokens + desc_tokens + tag_tokens]
        video["topic"] = determine_topic(topic_tokens)
        
        processed_videos.append(video)
        
    return processed_videos

async def process_all_csvs(csv_files, processed_videos):
    for csv_file in csv_files:
        logging.info("Processing %s", csv_file)
        os.makedirs("data/thumbnails", exist_ok=True)
        
        videos = await read_csv(csv_file)
        logging.info("Processed %s videos from CSV %s", len(videos), csv_file)
        processed_videos.extend(videos)

if __name__ == "__main__":
    DIRECTORY = "data/raw"
    
    processed_videos = []
    
    csv_files = [os.path.join(DIRECTORY, file) for file in os.listdir(DIRECTORY) if file.endswith(".csv")]
    asyncio.run(process_all_csvs(csv_files, processed_videos))

    logging.info("Processed %s videos in total.", len(processed_videos))
    
    if processed_videos:
        processed_df = pd.DataFrame(processed_videos)
        processed_df.to_csv("data/processed.csv", index=False)
        