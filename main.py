from db_setup import get_collection
from ingest import load_model, get_image_paths, embed_images, store_in_chroma 
from config import RECEIPTS_FOLDER


def main():

    print("Connecting to chroma...")
    collection = get_collection() 


    print("Loading model...")
    model = load_model() 


    print("Scanning for receipt images...")
    image_paths = get_image_paths(RECEIPTS_FOLDER)
    print(f"Found {len(image_paths)} images.")


    if not image_paths:
        print("No images found. Exiting.")
        return
    

    print("Embedding images...")
    embeddings = embed_images(model, image_paths)


    print("Storing in chroma...")
    store_in_chroma(collection, image_paths, embeddings)

    print(f"Done. collection now has {collection.count()} items.")


if __name__== "__main__":
    main()