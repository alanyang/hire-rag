import os
from openai import OpenAI
from chromadb import PersistentClient


product = """
1.  **產品名稱和承保機構**：明確是哪個產品由誰承保。
2.  **核心目的/價值主張**：產品主要解決什麼問題，為客戶帶來什麼利益。
3.  **主要特點**：
    *   供款期和保障期（2年供款，5年保障）。
    *   期滿金額的性質（保證）。
    *   身故賠償的計算方式（關鍵數字，如102%或保證現金價值）。
    *   額外意外身故賠償的觸發條件和上限。
    *   投保條件（年齡、貨幣、最低保費）。
    *   是否需要體檢。
4.  **運作方式**：
    *   預繳保費的具體規則。
    *   保單貸款的影響。
5.  **風險提示**：雖然您要求刪除「聲明」等，但產品固有的、對客戶決策有重大影響的風險（如信貸風險、匯率風險、提前退保損失）是「重要資訊」，應以簡潔方式提及。
6.  **重要提示**：關於保單文件、條款的最終約束力，以及建議客戶查閱詳細資料的提示，這屬於負責任的產品說明，也應保留。
    """

# product_embeddings = model.encode(product)


client = PersistentClient(path="/root/develop/rag/hire")

col = client.get_or_create_collection(name="hire")

# col.add(ids=["12"], documents=[product], embeddings=[product_embeddings.tolist()])

query = """
产品要求，保险产品。
"""

# query_embeddings = model.encode(query)

# r = col.query(query_embeddings=[query_embeddings.tolist()], n_results=3)
# print(r)


llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
response = llm.embeddings.create(input=product, model="text-embedding-3-small")

print(len(response.data[0].embedding))
