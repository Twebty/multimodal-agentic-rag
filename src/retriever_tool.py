from langchain_core.tools import tool

from retrieval import search_multimodal


# ==========================================
# TOOL 1 : RECHERCHE TEXTE
# ==========================================

@tool
def text_retriever_tool(query: str) -> str:
    """
    Recherche des informations textuelles et visuelles
    dans le document PDF 'Attention is all you need'.

    À utiliser lorsqu'une question textuelle nécessite
    des informations provenant du document.
    """

    results = search_multimodal(
        query=query,
        query_type="text",
        k=5
    )

    return format_results(results)


# ==========================================
# TOOL 2 : RECHERCHE IMAGE
# ==========================================

@tool
def image_retriever_tool(image_path: str) -> str:
    """
    Recherche les contenus les plus similaires à une image
    fournie par son chemin local.

    L'image est encodée avec CLIP puis comparée
    aux textes et images présents dans ChromaDB.
    """

    results = search_multimodal(
        query=image_path,
        query_type="image",
        k=5
    )

    return format_results(results)


# ==========================================
# FORMATAGE
# ==========================================

def format_results(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []

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

        similarity = 1 - distance

        content_type = metadata.get(
            "content_type"
        )

        page = metadata.get(
            "page"
        )

        if content_type == "text":

            output.append(
                f"""
RESULTAT {i}
Type : TEXT
Page : {page}
Similarité : {similarity:.4f}

Contenu :
{document}
"""
            )

        elif content_type == "image":

            image_path = metadata.get(
                "image_path"
            )

            output.append(
                f"""
RESULTAT {i}
Type : IMAGE
Page : {page}
Similarité : {similarity:.4f}
Image : {image_path}
"""
            )

    if not output:

        return "Aucun résultat pertinent trouvé."

    return "\n".join(output)


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEST DES TOOLS")
    print("=" * 70)

    print("\nTools disponibles :")

    print(
        "\n1. text_retriever_tool"
    )

    print(
        "2. image_retriever_tool"
    )

    choice = input(
        "\nChoisissez 1 ou 2 : "
    ).strip()

    if choice == "1":

        question = input(
            "\nQuestion : "
        )

        result = text_retriever_tool.invoke(
            {
                "query": question
            }
        )

        print("\n")
        print(result)

    elif choice == "2":

        image_path = input(
            "\nChemin de l'image : "
        )

        result = image_retriever_tool.invoke(
            {
                "image_path": image_path
            }
        )

        print("\n")
        print(result)

    else:

        print(
            "Choix invalide."
        )