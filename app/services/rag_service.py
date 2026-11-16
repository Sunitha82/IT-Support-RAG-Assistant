"""RAG orchestration with PII protection and optional metadata-only tracing."""
from dataclasses import dataclass, field, replace
from app.llm.azure_client import AzureLLMClient
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import SemanticRetriever
from app.guardrails.input_guard import validate_question
from app.guardrails.pii_guard import mask_pii
from app.observability.langfuse_tracing import observation
from app.core.config import get_settings
from app.retrieval.enhanced_retriever import EnhancedRetriever

NO_EVIDENCE_MESSAGE = "I don't have enough information in the knowledge base to answer that question."

@dataclass
class RetrievalDetail:
    source: str
    chunk_id: str
    rank: int
    distance: float
    similarity: float
    content: str
    child_id: str | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    retrieval_details: list[RetrievalDetail] = field(default_factory=list)
    privacy: dict = field(default_factory=dict)
    trace_id: str | None = None
    tracing_status: str = "disabled_or_unavailable"

class RAGService:
    def __init__(self, retriever=None, llm_client=None):
        if retriever is not None:
            self.retriever = retriever
        elif get_settings().retrieval_mode == "enhanced":
            self.retriever = EnhancedRetriever()
        else:
            self.retriever = SemanticRetriever()
        self.llm_client = llm_client if llm_client is not None else AzureLLMClient()

    def ask(self, question: str, top_k: int = 5) -> RAGResponse:
        with observation("input-guardrail", as_type="guardrail") as span:
            question = validate_question(question)
            span.update(output={"status": "accepted"})
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer between 1 and 10")
        with observation("input-pii-masking", as_type="guardrail") as span:
            question, input_count = mask_pii(question)
            span.update(output={"detections": input_count})
        question = question.strip()
        with observation("retrieval", as_type="retriever", input={"top_k": top_k}) as span:
            results = self.retriever.retrieve(question, top_k=top_k)
            span.update(output={"returned_sections": len(results)})
        privacy = {"input_detections": input_count, "context_detections": 0,
                   "output_detections": 0, "input_checked": True,
                   "context_checked": True, "output_checked": False}
        if not results:
            return RAGResponse(question=question, answer=NO_EVIDENCE_MESSAGE, privacy=privacy)
        safe_results = []
        with observation("context-pii-masking", as_type="guardrail") as span:
            for result in results:
                content, count = mask_pii(result.content)
                source, source_count = mask_pii(result.source)
                chunk_id, id_count = mask_pii(result.chunk_id)
                child_id, child_count = mask_pii(result.child_id) if result.child_id else (None, 0)
                privacy["context_detections"] += count + source_count + id_count + child_count
                safe_results.append(replace(result, content=content, source=source,
                                            chunk_id=chunk_id, child_id=child_id))
            span.update(output={"detections": privacy["context_detections"]})
        details = [RetrievalDetail(
            source=result.source, chunk_id=result.chunk_id, rank=rank,
            distance=round(float(result.distance), 4),
            similarity=round(1.0 - float(result.distance), 4), content=result.content,
            child_id=result.child_id, rerank_score=result.rerank_score, rrf_score=result.rrf_score,
        ) for rank, result in enumerate(safe_results, start=1)]
        with observation("azure-openai-generation", as_type="generation",
                         model=get_settings().azure_openai_deployment,
                         input={"context_sections": len(safe_results)}) as span:
            answer = self.llm_client.generate(
                system_prompt=SYSTEM_PROMPT, user_prompt=build_user_prompt(question, safe_results))
            span.update(output={"status": "completed"})
        with observation("output-pii-masking", as_type="guardrail") as span:
            answer, privacy["output_detections"] = mask_pii(answer)
            privacy["output_checked"] = True
            span.update(output={"detections": privacy["output_detections"]})
        return RAGResponse(question=question, answer=answer,
                           sources=list(dict.fromkeys(r.source for r in safe_results)),
                           retrieval_details=details, privacy=privacy)

    def ask_with_tracing(self, question: str, top_k: int = 5):
        with observation("it-support-rag-request", input={"top_k": top_k}) as span:
            response = self.ask(question, top_k=top_k)
            response.trace_id = span.trace_id
            response.tracing_status = "queued" if response.trace_id else "disabled_or_unavailable"
            span.update(output={"status": "answered" if response.sources else "no_evidence"},
                        metadata={**response.privacy, "source_count": len(response.sources)})
            return response
