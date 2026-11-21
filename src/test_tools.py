
from retriever_tool import (
    text_retriever_tool,
    image_retriever_tool
)

# ==================================
# TEST 1 : Recherche textuelle
# ==================================

print("\n========== TEST 1 : TEXT RETRIEVER ==========")

result_text = text_retriever_tool.invoke({
    "query": "What is the Transformer architecture?"
})

print(result_text)


# ==================================
# TEST 2 : Recherche par image
# ==================================

print("\n========== TEST 2 : IMAGE RETRIEVER ==========")

result_image = image_retriever_tool.invoke({
    "image_path": "images/page_3_image_1.png"
})

print(result_image)
