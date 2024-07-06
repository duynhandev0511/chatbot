import cohere
import uuid
import docx
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

COHERE_API_KEY = "Your api key"
co = cohere.Client(COHERE_API_KEY)

raw_documents = [
    {
        "title": "Các nguyên tố hóa học",
        "url": "E:\\bot\\documents\\bangtuanhoan_db.docx"
    },
    {
        "title": "SGK Cánh Diều Hóa 12",
        "url": "E:\\bot\\documents\\Hóa 12 - CD.SGK.docx"
    },
    {
        "title": "SGK Chân trời sáng tạo Hóa 12",
        "url": "E:\\bot\\documents\\Hoa 12 - CTST.SGK.docx"
    },
]

class Vectorstore:
    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the HTML content or .doc files.
        """
        print("Loading documents...")

        for raw_document in self.raw_documents:
            if raw_document["url"].endswith(".doc") or raw_document["url"].endswith(".docx"):
                text = self.extract_text_from_doc(raw_document["url"])
                chunks = self.chunk_text(text)
                for chunk in chunks:
                    self.docs.append(
                        {
                            "title": raw_document["title"],
                            "text": chunk,
                            "url": raw_document["url"],
                        }
                    )
            else:
                elements = partition_html(url=raw_document["url"])
                chunks = chunk_by_title(elements)
                for chunk in chunks:
                    self.docs.append(
                        {
                            "title": raw_document["title"],
                            "text": str(chunk),
                            "url": raw_document["url"],
                        }
                    )

    def extract_text_from_doc(self, file_path: str) -> str:
        """
        Extracts text from a .doc or .docx file.
        
        Parameters:
        file_path (str): Path to the .doc or .docx file.
        
        Returns:
        str: Extracted text.
        """
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunks a given text into smaller pieces.
        
        Parameters:
        text (str): Text to chunk.
        
        Returns:
        List[str]: List of text chunks.
        """
        # Implement your chunking logic here. For simplicity, let's split by paragraphs.
        return text.split("\n\n")

    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """
        print("Embedding document chunks...")

        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.idx.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        Parameters:
        query (str): The query to retrieve document chunks for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """

        # Dense retrieval
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        # Reranking
        rank_fields = ["title", "text"]  # We'll use the title and text fields for reranking

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )

        docs_retrieved = []
        for doc_id in doc_ids:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved



    def is_relevant(self, message: str, documents: List[Dict[str, str]]) -> bool:
            """
            Kiểm tra xem các tài liệu có liên quan đến hóa học hay không.
            
            Parameters:
            message (str): Câu hỏi của người dùng.
            documents (List[Dict[str, str]]): Các tài liệu được truy xuất.
            
            Returns:
            bool: True nếu có liên quan đến hóa học, ngược lại False.
            """
            # Tách các từ khóa từ câu hỏi và loại bỏ các từ không cần thiết
            keywords = [keyword.lower() for keyword in message.split() if keyword.lower() not in ['và', 'hoặc', 'là', 'có']]

            # Lọc các tài liệu dựa trên từ khóa
            for doc in documents:
                # Chuyển đổi văn bản tài liệu và các từ khóa về chữ thường để so sánh
                doc_text_lower = doc["text"].lower()
                if any(keyword in doc_text_lower for keyword in keywords):
                    return True
            return False




class Chatbot:
    def __init__(self, vectorstore: Vectorstore):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.initial_greeting = "Chào bạn! Tôi có thể giúp gì cho bạn?"

    def preprocess_query(self, message: str) -> str:
        """
        Tiền xử lý câu hỏi để chỉ giữ lại các từ khóa quan trọng.

        Parameters:
        message (str): Câu hỏi từ người dùng.

        Returns:
        str: Câu hỏi đã được tiền xử lý.
        """
        keywords = [keyword.lower() for keyword in message.split() if keyword.lower() not in ['và', 'hoặc', 'là', 'có']]
        return " ".join(keywords)

    def run(self, message: str) -> dict:
        response = co.chat(message=message, search_queries_only=True)

        if response.search_queries:
            print("Đang truy xuất thông tin...")

            documents = []
            for query in response.search_queries:
                processed_query = self.preprocess_query(query.text)
                retrieved_docs = self.vectorstore.retrieve(processed_query)
                documents.extend(retrieved_docs)

            # Kiểm tra xem có tài liệu hóa học phù hợp không
            if not documents or not self.vectorstore.is_relevant(processed_query, documents):
                response_text = "Xin lỗi, tôi chỉ có thể trả lời những câu hỏi liên quan đến hóa học."
                return {"response": response_text}

            # Tiếp tục xử lý và trả lời dựa trên tài liệu đã lọc
            response = co.chat_stream(
                message=message,
                model="command-r",
                documents=documents,
                conversation_id=self.conversation_id,
            )

            response_text = ""
            for event in response:
                if event.event_type == "text-generation":
                    response_text += event.text
        else:
            response = co.chat_stream(
                message=message,
                model="command-r",
                conversation_id=self.conversation_id,
            )

            response_text = ""
            for event in response:
                if event.event_type == "text-generation":
                    response_text += event.text

        return {"response": response_text}

# Khởi tạo Vectorstore và Chatbot
vectorstore = Vectorstore(raw_documents)
chatbot = Chatbot(vectorstore)

# Nhận đầu vào từ người dùng và chạy chatbot
while True:
    message = input("Nhập câu hỏi của bạn: ")
    if message.lower() in ["exit", "quit"]:
        break
    response = chatbot.run(message)
    print(response["response"])

