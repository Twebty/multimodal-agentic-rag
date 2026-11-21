import os

import chromadb
import open_clip
import torch
from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "attention_multimodal"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 1. CHARGEMENT DE CLIP
# ==========================================

print(f"Appareil utilisé : {device}")
print("Chargement de CLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)

model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

print("CLIP chargé.")


# ==========================================
# 2. CONNEXION À CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"Collection chargée : {COLLECTION_NAME}"
)

print(
    f"Nombre d'éléments : {collection.count()}"
)


# ==========================================
# 3. EMBEDDING D'UNE REQUÊTE TEXTE
# ==========================================

def encode_text_query(query):

    tokens = tokenizer(
        [query]
    ).to(device)

    with torch.no_grad():

        embedding = model.encode_text(
            tokens
        )

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

    return (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
        .tolist()
    )


# ==========================================
# 4. EMBEDDING D'UNE REQUÊTE IMAGE
# ==========================================

def encode_image_query(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = preprocess(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        embedding = model.encode_image(
            image_tensor
        )

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

    return (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
        .tolist()
    )


# ==========================================
# 5. RECHERCHE VECTORIELLE
# ==========================================

def search_multimodal(
    query,
    query_type="text",
    k=5
):

    if query_type == "text":

        query_embedding = encode_text_query(
            query
        )

    elif query_type == "image":

        query_embedding = encode_image_query(
            query
        )

    else:

        raise ValueError(
            "query_type doit être 'text' ou 'image'."
        )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


# ==========================================
# 6. AFFICHAGE DES RÉSULTATS
# ==========================================

def display_results(results):

    print("\n" + "=" * 70)
    print("RÉSULTATS DE LA RECHERCHE")
    print("=" * 70)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances
        ),
        start=1
    ):

        # Pour une distance cosinus Chroma :
        # similarité ≈ 1 - distance
        similarity = 1 - distance

        print(f"\nRésultat #{i}")

        print(
            f"Type : "
            f"{metadata.get('content_type')}"
        )

        print(
            f"Page : "
            f"{metadata.get('page')}"
        )

        print(
            f"Similarité : "
            f"{similarity:.4f}"
        )

        if metadata.get(
            "content_type"
        ) == "text":

            preview = document[:500]

            print(
                f"Texte : {preview}"
            )

        elif metadata.get(
            "content_type"
        ) == "image":

            print(
                "Image :",
                metadata.get(
                    "image_path"
                )
            )


# ==========================================
# 7. TEST INTERACTIF
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("RECHERCHE MULTIMODALE")
    print("=" * 70)

    print("\n1 - Requête texte")
    print("2 - Requête image")

    choice = input(
        "\nChoisissez 1 ou 2 : "
    ).strip()

    if choice == "1":

        query = input(
            "Votre question : "
        ).strip()

        results = search_multimodal(
            query=query,
            query_type="text",
            k=5
        )

        display_results(
            results
        )

    elif choice == "2":

        image_path = input(
            "Chemin de l'image : "
        ).strip()

        results = search_multimodal(
            query=image_path,
            query_type="image",
            k=5
        )

        display_results(
            results
        )

    else:

        print(
            "Choix invalide."
        )