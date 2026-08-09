from sentence_transformers import SentenceTransformer
import os
import time
import numpy as np
from typing import Optional

# NOTE : Library Hierarchy represents the flow of the data
from RagPipeline.tools.pdf_extractor import ExtractPdfContent
from RagPipeline.tools.chunking import CreateChunking
from RagPipeline.tools.create_embeddings import create_embedding
from RagPipeline.tools.ranker import build_faiss_index
from RagPipeline.tools.retrieve_top_k import retrieve_top_k
from RagPipeline.tools.generator import create_generator
            
class RagPipelineConnector:
    """
    RAG pipeline using FAISS retrieval + optional LLM generation.
    Set generate_answer=True in query() to use the local generator model.
    
    if **Doc path** is provided then dont use the **pdf_text** argument and for **pdf_text** dont use the **doc_path**
    
    
    """

    def __init__(
        self,
        doc_path:Optional[str] = None,
        pdf_text:Optional[str] = None,
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embd_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        llm_device: str = "cpu",
    ):
        super().__init__()

        self.doc_path = doc_path
        self.doc_text = pdf_text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.device = device if device != "auto" else "cpu"
        self.embd_model_name = embd_model_name
        self.llm_model_name = llm_model_name
        self.llm_device = llm_device

        # Step 1: Extract document
        if self.doc_path is not None:
            ext = os.path.splitext(doc_path)[1].lower()
            t0 = time.time()
            self.pdf_text = ExtractPdfContent().extract_texts(pdf_url=self.doc_path)
            print(f"[{time.time() - t0:.1f}s] {ext[1:]} extracted successfully — "
                  f"{len(self.pdf_text['text'])} words")

            # Step 2: Chunk
            t0 = time.time()
            chunk_worker = CreateChunking()
            self.chunks = chunk_worker.create_chunk(
                file_path=self.doc_path,
                doc_text=self.pdf_text['text'],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            print(f"[{time.time() - t0:.1f}s] {len(self.chunks)} chunks created")

        else:
            chunking_time_start = time.time()
            chunk_worker = CreateChunking()
            self.chunks = chunk_worker.create_chunk(
                doc_text=self.doc_text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            
            print(f"[{time.time() - chunking_time_start:.1f}s] {len(self.chunks)} chunks created")
            
            
        # Step 3: Embed
        t0 = time.time()
        self.embd_model = SentenceTransformer(self.embd_model_name, device=self.device)
        self.chunks_embeddings = create_embedding(
            model=self.embd_model, chunks=self.chunks
        )
        print(f"[{time.time() - t0:.1f}s] Embeddings created — "
              f"shape={self.chunks_embeddings.shape}")

        # Step 4: FAISS index
        t0 = time.time()
        self.faiss_index = build_faiss_index(
            chunk_embeddings=self.chunks_embeddings
        )
        print(f"[{time.time() - t0:.1f}s] FAISS index built — "
              f"{self.faiss_index.ntotal} vectors")

        print(f"Pipeline ready. top_k={self.top_k}, device={self.device}")

    def query(self, question: str, generate_answer: bool = True) -> dict:
        """
        Retrieve top-k chunks from FAISS. Optionally generate an LLM answer.

        Parameters
        ----------
        question : str
            User question.
        generate_answer : bool
            If True, uses the local generator model to produce an answer
            from the retrieved context.

        Returns
        -------
        dict with keys: question, answer, source_chunks, total_sources
        """
        # Retrieve using FAISS
        retrieved = retrieve_top_k(
            question=question,
            embedding_model=self.embd_model,
            faiss_index=self.faiss_index,
            chunks_data=self.chunks,
            top_k=self.top_k,
        )

        # Build source chunks list with full text
        source_chunks = [
            {
                "id": r["chunk_id"],
                "text": r["chunk_text"],
            }
            for r in retrieved
        ]

        # Generate answer using local LLM (optional)
        if generate_answer:
            answer = create_generator(
                query=question,
                context=retrieved,
                model_name=self.llm_model_name,
                device=self.llm_device,
            )
        else:
            answer = ""

        return {
            "question": question,
            "answer": answer,
            "source_chunks": source_chunks,
            "total_sources": len(source_chunks),
        }