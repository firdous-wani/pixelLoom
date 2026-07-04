
from db_setup import get_collection
from ingest import load_model

def embed_query(model, query_text):
    embedding = model.encode_query([query_text])
    return embedding

def search_receipts(collection, query_embedding, n_results=3):
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results

def print_results(results):
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for rank, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
        print(f"{rank}. {meta['file_path']}  (distance: {dist:.4f})")

def main():
    collection = get_collection()
    model = load_model()

    query_text = input("What receipt are you looking for? ")
    query_embedding = embed_query(model, query_text)

    results = search_receipts(collection, query_embedding)
    print_results(results)

if __name__ == "__main__":
    main()