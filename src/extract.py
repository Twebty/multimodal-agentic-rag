import fitz
import os
import json

PDF_PATH = "data/Attention is all you need.pdf"
IMAGES_DIR = "images"
OUTPUT_TEXT = "data/extracted_text.json"

os.makedirs(IMAGES_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)

pages_data = []

print(f"Nombre de pages : {len(doc)}")

for page_number in range(len(doc)):
    page = doc.load_page(page_number)

    # =========================
    # 1. EXTRACTION DU TEXTE
    # =========================

    text = page.get_text("text")

    pages_data.append({
        "page": page_number + 1,
        "text": text
    })

    print(
        f"Page {page_number + 1}: "
        f"{len(text)} caractères extraits"
    )

    # =========================
    # 2. EXTRACTION DES IMAGES
    # =========================

    images = page.get_images(full=True)

    for image_index, img in enumerate(images):
        xref = img[0]

        base_image = doc.extract_image(xref)

        image_bytes = base_image["image"]
        image_extension = base_image["ext"]

        image_name = (
            f"page_{page_number + 1}_"
            f"image_{image_index + 1}."
            f"{image_extension}"
        )

        image_path = os.path.join(
            IMAGES_DIR,
            image_name
        )

        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)

        print(f"   Image sauvegardée : {image_path}")

# Sauvegarde du texte
with open(
    OUTPUT_TEXT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        pages_data,
        f,
        ensure_ascii=False,
        indent=4
    )

doc.close()

print("\nExtraction terminée.")
print(f"Texte sauvegardé dans : {OUTPUT_TEXT}")
print(f"Images sauvegardées dans : {IMAGES_DIR}")