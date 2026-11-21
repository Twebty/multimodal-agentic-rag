
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from retriever_tool import (
    text_retriever_tool,
    image_retriever_tool
)


# ==========================================
# 1. CHARGEMENT DE LA CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY introuvable. "
        "Vérifiez votre fichier .env."
    )


# ==========================================
# 2. CONFIGURATION DU LLM (MIND)
# ==========================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)


# ==========================================
# 3. CONFIGURATION DES TOOLS (BODY)
# ==========================================

tools = [
    text_retriever_tool,
    image_retriever_tool
]


# ==========================================
# 4. PROMPT SYSTEME
# ==========================================

SYSTEM_PROMPT = """
Tu es un assistant intelligent spécialisé
dans le document scientifique :

Attention is all you need.

Ton rôle est de répondre aux questions
en utilisant les informations du document.

REGLES :

1. Utilise text_retriever_tool pour rechercher
   les informations dans le PDF.

2. Utilise image_retriever_tool lorsqu'un
   utilisateur fournit un chemin d'image local
   à analyser et comparer avec le document.

3. Tu peux appeler plusieurs outils si nécessaire.

4. Analyse les observations retournées par
   les outils avant de répondre.

5. Cite les numéros des pages lorsqu'ils sont
   disponibles.

6. Ne fabrique jamais une information absente
   des résultats.

7. Si les informations sont insuffisantes,
   indique clairement cette limite.

8. Réponds en français, avec des explications
   simples et structurées.
"""


# ==========================================
# 5. CONSTRUCTION DE L'AGENT
# ==========================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    name="multimodal_rag_agent"
)


# ==========================================
# 6. EXECUTION ET OBSERVATION REACT
# ==========================================

def run_agent(question):

    print("\n" + "=" * 70)
    print("QUESTION UTILISATEUR")
    print("=" * 70)

    print(question)

    inputs = {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }

    print("\n" + "=" * 70)
    print("EXECUTION DU GRAPHE LANGGRAPH")
    print("=" * 70)

    final_answer = None

    # Observation des différentes étapes du graphe
    for update in agent.stream(
        inputs,
        stream_mode="updates"
    ):

        for node_name, node_data in update.items():

            print(f"\n[NODE] {node_name}")

            messages = node_data.get("messages", [])

            for message in messages:

                message_type = type(message).__name__

                print("Type :", message_type)

                # Décision du LLM d'appeler un outil
                if getattr(message, "tool_calls", None):

                    for call in message.tool_calls:

                        print(
                            "ACTION :",
                            call["name"]
                        )

                        print(
                            "ARGUMENTS :",
                            call["args"]
                        )

                # Observation retournée par un outil
                elif message_type == "ToolMessage":

                    print("OBSERVATION :")
                    print(str(message.content)[:1000])

                # Réponse finale du LLM
                elif message_type == "AIMessage":

                    if message.content:
                        final_answer = message.content

    print("\n" + "=" * 70)
    print("REPONSE FINALE DE L'AGENT")
    print("=" * 70)

    print(final_answer)

    return final_answer


# ==========================================
# 7. INTERFACE TERMINAL
# ==========================================

if __name__ == "__main__":

    print("\nMULTIMODAL RAG AGENT")

    print("Document : Attention is all you need")

    print("Tapez 'exit' pour quitter.")

    while True:

        question = input(
            "\nVotre question : "
        ).strip()

        if question.lower() in ["exit", "quit"]:

            print("Fermeture de l'agent.")

            break

        if not question:
            continue

        try:

            run_agent(question)

        except Exception as error:

            print("\nErreur :", error)
