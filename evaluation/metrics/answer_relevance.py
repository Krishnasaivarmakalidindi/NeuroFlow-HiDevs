import re
import logging
from opentelemetry import trace

try:
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria
    from providers.base import ChatMessage
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria
    from providers.base import ChatMessage

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

def cosine_similarity(v1: list, v2: list) -> float:
    dot_prod = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(a * a for a in v2) ** 0.5
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_prod / (norm_v1 * norm_v2)

async def evaluate_answer_relevance(
    query: str,
    answer: str,
    **kwargs
) -> float:
    client = NeuroFlowClient()
    criteria = RoutingCriteria(task_type="evaluation")

    with tracer.start_as_current_span("evaluation.answer_relevance") as span:
        prompt = (
            "Generate 3-5 questions that this answer answers.\n\n"
            f"Answer:\n{answer}"
        )

        try:
            res = await client.chat([ChatMessage(role="user", content=prompt)], criteria, **kwargs)
            content = res.content
        except Exception as e:
            logger.error(f"Failed to generate questions in answer_relevance: {e}")
            span.record_exception(e)
            span.set_attribute("score", 0.0)
            return 0.0

        # Parse questions
        lines = [line.strip() for line in content.split('\n')]
        generated_questions = []
        for line in lines:
            # Strip numberings, bullet points, leading/trailing whitespaces
            cleaned = re.sub(r'^(?:\d+[\.\)\-]|[\-*•])\s*', '', line).strip()
            if cleaned:
                generated_questions.append(cleaned)

        # Slice to 3-5 questions
        generated_questions = [q for q in generated_questions if q][:5]
        if len(generated_questions) < 3:
            # Fallback if too few questions generated
            generated_questions.append(query)

        # Embed all texts: query + generated questions
        all_texts = [query] + generated_questions
        try:
            embeddings = await client.embed(all_texts, criteria)
        except Exception as e:
            logger.error(f"Failed to generate embeddings in answer_relevance: {e}")
            span.record_exception(e)
            span.set_attribute("score", 0.0)
            return 0.0

        query_vector = embeddings[0]
        question_vectors = embeddings[1:]

        similarities = [cosine_similarity(query_vector, q_vec) for q_vec in question_vectors]
        
        if not similarities:
            score = 0.0
        else:
            score = sum(similarities) / len(similarities)

        span.set_attribute("score", score)
        span.set_attribute("generated_questions_count", len(generated_questions))
        return score
