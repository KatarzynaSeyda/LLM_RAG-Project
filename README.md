# 📑 Analiza umów publishingowycn z wykorzystaniem RAG

Projekt ma na celu stworzenie systemu wykorzystującego podejście Retrieval-Augmented Generation (RAG) do przeszukiwania i analizy dokumentów publishingowych.

Projekt zawiera następujące elementy:

1. 🗂 Integracja danych: Zgromadzenie zbioru umów publishingowych w formacie docx. Odpowiednie przygotowanie ich w podziale na zdania.
2. 📐Przygotowanie wektorów o najmniejszej odległości cosinusowej:
      - Pierwszy etap to określenie, z którego dokumentu/umowy narzędzie powinno szukać odpowiedzi (Document-Level Embedding)
      - Drugi etap to określenie zdań z wybranej umowy, które będą źródłem odpowiedzi (Sentence-Level Embeddings)
      - Wykorzystano model **all-mpnet-base-v2**, który jest odpowiedni do wyszukiwania informacji i analizy dokumentów prawnych, umów etc.

3. 💬 Implementacja RAG wykorzustując model LLM z bibloteki 🦙**ollama**:
   - Określono kontekst i rolę, która pozwala na lepsze dopasowanie odpowiedzi: *'You are a financial controller. Answer the question using the provided context from legal agreements. Keep it concise, briefly. If you don't know the answer let me know.*
   - Wykorzystano model **initium/law_model**