from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

# Router oluştur
router = APIRouter(
    prefix="/api/v1",
    tags=["rag"]
)

# Veri modelleri
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class DocumentInfo(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    response: str
    relevant_documents: List[DocumentInfo]

# Bağımlılık enjeksiyonu için RAG instance'ı
def get_rag_instance():
    from app.core.rag import get_rag_instance
    rag = get_rag_instance()
    if not rag:
        raise HTTPException(status_code=500, detail="RAG sistemi henüz başlatılmadı")
    return rag

# Sorgulama endpoint'i
@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag=Depends(get_rag_instance)):
    try:
        result = rag.query(request.query, top_k=request.top_k)
        
        # Sonuçları Pydantic modeline dönüştür
        response = QueryResponse(
            response=result["response"],
            relevant_documents=[
                DocumentInfo(
                    text=doc["text"],
                    score=doc["score"],
                    metadata=doc["metadata"]
                ) for doc in result["relevant_documents"]
            ]
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorgulama hatası: {str(e)}")

# Durum endpoint'i
@router.get("/status")
async def status(rag=Depends(get_rag_instance)):
    return {
        "status": "online",
        "model_loaded": True,
        "vector_db_documents": rag.get_document_count()
    } 