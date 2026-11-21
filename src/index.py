import hashlib
import os
import re

import chromadb
import torch


# ==========================================
# CONFIGURATION
# ==========================================

PDF_PATH = "data/Attention is all you need.pdf"

TEXT_EMBEDDINGS = "data/text_embeddings.pt"
IMAGE_EMBEDDINGS = "data/image_embeddings.pt"

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "attention_multimodal"


# ==========================================
# 1. CALCUL DU SHA-256
# ==========================================

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


document_id = calculate_sha256(PDF_PATH)

print("=" * 60)
print("INDEXATION MULTIMODALE")
print("=" * 60)

print("\nDocument :", PDF_PATH)
print("SHA-256 :", document_id)


# ==========================================
# 2. CHARGEMENT DES EMBEDDINGS
# ==========================================

print("\nChargement des embeddings...")

text_results = torch.load(
    TEXT_EMBEDDINGS,
    weights_only=False
)

image_results = torch.load(
    IMAGE_EMBEDDINGS,
    weights_only=False
)

print(
    f"Embeddings texte chargés : {len(text_results)}"
)

print(
    f"Embeddings image chargés : {len(image_results)}"
)


# ==========================================
# 3. INITIALISATION CHROMADB
# ==========================================

print("\nInitialisation de ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)

print(
    f"Collection : {COLLECTION_NAME}"
)


# ==========================================
# 4. SUPPRESSION ANCIENNE INDEXATION
# ==========================================

# Cela permet d'éviter les doublons si nous
# réindexons exactement le même document.

try:
    collection.delete(
        where={
            "document_id": document_id
        }
    )

    print(
        "\nAncienne indexation du document supprimée."
    )

except Exception:
    pass


# ==========================================
# 5. INDEXATION DU TEXTE
# ==========================================

print("\nIndexation des textes...")

for index, item in enumerate(text_results):

    page = item["page"]
    text = item["text"]

    embedding = (
        item["embedding"]
        .squeeze(0)
        .numpy()
        .tolist()
    )

    item_id = (
        f"{document_id}_text_{page}_{index}"
    )

    collection.add(
        ids=[item_id],

        embeddings=[
            embedding
        ],

        documents=[
            text
        ],

        metadatas=[
            {
                "document_id": document_id,
                "content_type": "text",
                "page": int(page),
                "source": PDF_PATH
            }
        ]
    )

    print(
        f"Texte page {page} indexé."
    )


# ==========================================
# 6. EXTRAIRE LE NUMÉRO DE PAGE D'UNE IMAGE
# ==========================================

def extract_page_number(filename):

    match = re.search(
        r"page_(\d+)",
        filename
    )

    if match:
        return int(match.group(1))

    return -1


# ==========================================
# 7. INDEXATION DES IMAGES
# ==========================================

print("\nIndexation des images...")

for index, item in enumerate(image_results):

    path = item["path"]
    filename = item["filename"]

    page = extract_page_number(
        filename
    )

    embedding = (
        item["embedding"]
        .squeeze(0)
        .numpy()
        .tolist()
    )

    item_id = (
        f"{document_id}_image_{page}_{index}"
    )

    collection.add(
        ids=[item_id],

        embeddings=[
            embedding
        ],

        documents=[
            f"Image extraite de la page {page}"
        ],

        metadatas=[
            {
                "document_id": document_id,
                "content_type": "image",
                "page": int(page),
                "image_path": path,
                "source": PDF_PATH
            }
        ]
    )

    print(
        f"Image {filename} indexée "
        f"(page {page})."
    )


# ==========================================
# 8. VÉRIFICATION
# ==========================================

total = collection.count()

print("\n" + "=" * 60)

print("INDEXATION TERMINÉE")

print("=" * 60)

print(
    f"Textes : {len(text_results)}"
)

print(
    f"Images : {len(image_results)}"
)

print(
    f"Total dans ChromaDB : {total}"
)

print(
    f"Base vectorielle : {CHROMA_PATH}"
)

print(
    f"Collection : {COLLECTION_NAME}"
)