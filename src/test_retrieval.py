
from retrieval import search_multimodal

question = "What is the Transformer architecture?"

print("\n========== TEST DE RECHERCHE TEXTUELLE ==========")

results = search_multimodal(
    query=question,
    query_type="text",
    k=5
)

print("\nRésultats obtenus :")

for i, result in enumerate(results, start=1):
    print(f"\nRésultat {i}")
    print(result)
# Image extraite du PDF
image_path = "images/page_3_image_1.png"

print("\n========== TEST DE RECHERCHE PAR IMAGE ==========")

results = search_multimodal(
    query=image_path,
    query_type="image",
    k=5
)

print("\nRésultats obtenus :")

for i, result in enumerate(results, start=1):
    print(f"\nRésultat {i}")
    print(result)
