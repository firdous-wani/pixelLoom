import os
from config import  EMBEDDINGS_MODEL_NAME
from sentence_transformers import SentenceTransformer
import hashlib

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".heic")

def load_model():
    model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
    return model 

def get_image_paths(folder):
  
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    paths = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(root, filename)
                paths.append(full_path)
    return paths


def embed_images(model, image_paths):
   
   
    documents = [{"image": path} for path in image_paths]
    embeddings = model.encode_document(documents)

    return embeddings


def make_id(path):

    return hashlib.md5(path.encode()).hexdigest()



def store_in_chroma(collection, image_paths, embeddings):
    
 
    ids = [make_id(path) for path in image_paths]
    metadatas  = [{"file_path": path} for path in image_paths]

    embeddings_list = embeddings.tolist() 

    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas
    )