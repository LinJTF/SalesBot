import argparse
from collections.abc import Sequence
from pathlib import Path

from vector_store.embedding import generate_openai_embeddings
from vector_store.ingestion import upload_products_to_qdrant
from vector_store.loader import format_products_for_embedding
from vector_store.loader import load_csv_products
from vector_store.qdrant_client import collection_exists
from vector_store.qdrant_client import collection_has_data
from vector_store.qdrant_client import create_collection
from vector_store.qdrant_client import setup_qdrant_client


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Populate Qdrant vector store with product data"
    )
    ap.add_argument(
        "--collection-name",
        default="products_promotions",
        help="Name of the Qdrant collection to create/use",
    )
    ap.add_argument(
        "--csv-path",
        default="docs/rag.csv",
        help="Path to the CSV file containing product data",
    )
    ap.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing even if collection already has data",
    )

    args = ap.parse_args(argv)

    if not Path(args.csv_path).exists():
        print(f"❌ Error: CSV file not found at {args.csv_path}")
        return 1

    client = setup_qdrant_client()

    # Check if collection exists and has data
    if not args.force_reindex and collection_has_data(client, args.collection_name):
        print("✅ Collection already exists with data. Skipping indexing to save time.")
        print("💡 Use --force-reindex to reindex anyway.")
        print(f"📦 Collection '{args.collection_name}' is ready for use.")
        return 0

    print("🔄 Proceeding with indexing...")

    products = load_csv_products(args.csv_path)
    texts = format_products_for_embedding(products)
    
    print("🔢 Generating embeddings...")
    embeddings = generate_openai_embeddings(texts)

    if not collection_exists(client, args.collection_name):
        create_collection(client, args.collection_name, vector_size=len(embeddings[0]))
    else:
        print(f"📦 Using existing collection '{args.collection_name}'")

    print("⬆️ Uploading vectors to Qdrant...")
    upload_products_to_qdrant(client, args.collection_name, products, embeddings)
    print(f"✅ Successfully uploaded {len(products)} product vectors.")

    print(f"📦 Collection '{args.collection_name}' is ready for use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
