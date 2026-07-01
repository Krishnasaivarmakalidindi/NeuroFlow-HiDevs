import logging
import uuid
import math
from typing import List, Dict, Any
from db.pool import DatabasePool

logger = logging.getLogger(__name__)

class QualityAnomalyDetector:
    async def detect_anomalies(self, pipeline_id: str) -> List[Dict[str, Any]]:
        try:
            pipeline_uuid = uuid.UUID(pipeline_id)
        except ValueError:
            return []

        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            # Fetch evaluations for the last 7 days
            query = """
                SELECT e.overall_score, e.faithfulness, e.answer_relevance, e.context_precision, e.context_recall, r.created_at
                FROM evaluations e
                JOIN pipeline_runs r ON e.run_id = r.id
                WHERE r.pipeline_id = $1 AND r.created_at >= NOW() - INTERVAL '7 days'
                ORDER BY r.created_at DESC;
            """
            try:
                rows = await conn.fetch(query, pipeline_uuid)
            except Exception as e:
                logger.error(f"Failed to query evaluations for anomaly detection: {e}")
                return []
            
        if len(rows) < 3:
            # Not enough data points to compute standard deviation
            return []

        scores = [float(r["overall_score"]) for r in rows if r["overall_score"] is not None]
        if len(scores) < 3:
            return []

        # Check the most recent run for anomaly against historical distribution (scores[1:])
        recent_score = scores[0]
        historical_scores = scores[1:]
        
        if len(historical_scores) < 2:
            return []

        mean = sum(historical_scores) / len(historical_scores)
        variance = sum((x - mean) ** 2 for x in historical_scores) / len(historical_scores)
        std_dev = math.sqrt(variance)

        # Standard threshold: mean - 2 * std_dev
        threshold = mean - 2 * std_dev
        
        # Avoid flagging tiny deviations by enforcing a minimal std_dev threshold of 0.005
        effective_std = max(std_dev, 0.005)
        threshold = mean - 2 * effective_std
        
        anomalies = []
        if recent_score < threshold:
            anomalies.append({
                "metric": "overall_score",
                "problem": f"Quality anomaly detected: recent run overall score ({recent_score:.4f}) is lower than 2 standard deviations below the 7-day average (mean: {mean:.4f}, std: {std_dev:.4f}, threshold: {threshold:.4f}).",
                "suggestion": "Investigate if recent document ingestion introduced low-quality chunks, or if there is drift in the LLM router's completions."
            })
            
        return anomalies
