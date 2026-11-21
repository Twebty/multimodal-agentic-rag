import json
import os

import open_clip
import torch
from PIL import Image


# ==============================
# CONFIGURATION
# ==============================

TEXT_FILE = "data/extracted_text.json"
IMAGES_DIR = "images"

OUTPUT_TEXT = "data/text_embeddings.pt"
OUTPUT_IMAGES = "data/image_embeddings.pt"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


# ==============================
# CPU OU GPU
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Appareil utilisé : {device}")


# ==============================
# CHARGEMENT DE CLIP
# ==============================

print("Chargement du modèle CLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)

model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

print("CLIP chargé avec succès.")


# ==============================
# EMBEDDINGS TEXTE
# ==============================

print("\nCréation des embeddings texte...")

with open(TEXT_FILE, "r", encoding="utf-8") as f:
    pages = json.load(f)


text_results = []

for page in pages:

    page_number = page["page"]
    text = page["text"].strip()

    if not text:
        continue

    # CLIP possède une limite de contexte assez courte.
    # On utilise ici un fragment raisonnable pour le TP.
    text = text[:1000]

    tokens = tokenizer([text]).to(device)

    with torch.no_grad():

        embedding = model.encode_text(tokens)

        # Normalisation L2
        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

    text_results.append({
        "page": page_number,
        "text": text,
        "embedding": embedding.cpu()
    })

    print(
        f"Page {page_number} : "
        f"embedding {embedding.shape}"
    )


torch.save(text_results, OUTPUT_TEXT)

print(
    f"\nEmbeddings texte sauvegardés : "
    f"{OUTPUT_TEXT}"
)


# ==============================
# EMBEDDINGS IMAGES
# ==============================

print("\nCréation des embeddings images...")


image_results = []


for filename in os.listdir(IMAGES_DIR):

    path = os.path.join(
        IMAGES_DIR,
        filename
    )

    try:

        image = Image.open(path).convert("RGB")

        image_tensor = preprocess(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            embedding = model.encode_image(
                image_tensor
            )

            # Normalisation L2
            embedding = embedding / embedding.norm(
                dim=-1,
                keepdim=True
            )

        image_results.append({
            "path": path,
            "filename": filename,
            "embedding": embedding.cpu()
        })

        print(
            f"{filename} : "
            f"embedding {embedding.shape}"
        )

    except Exception as e:

        print(
            f"Erreur avec {filename}: {e}"
        )


torch.save(
    image_results,
    OUTPUT_IMAGES
)


print(
    f"\nEmbeddings images sauvegardés : "
    f"{OUTPUT_IMAGES}"
)


# ==============================
# RÉSUMÉ
# ==============================

print("\n==============================")
print("VECTORISATION TERMINÉE")
print("==============================")

print(
    f"Textes vectorisés : "
    f"{len(text_results)}"
)

print(
    f"Images vectorisées : "
    f"{len(image_results)}"
)

if text_results:

    print(
        "Dimension embedding texte :",
        text_results[0]["embedding"].shape
    )

if image_results:

    print(
        "Dimension embedding image :",
        image_results[0]["embedding"].shape
    )