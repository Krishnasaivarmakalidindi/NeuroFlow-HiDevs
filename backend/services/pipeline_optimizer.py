import logging
from .pipeline_manager import PipelineManager
from .pipeline_analytics import PipelineAnalyticsService

logger = logging.getLogger(__name__)

class PipelineOptimizer:
    def __init__(self):
        self.manager = PipelineManager()
        self.analytics = PipelineAnalyticsService()

    async def get_suggestions(self, pipeline_id: str) -> list:
        # Fetch current config
        pipeline = await self.manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found.")

        config = pipeline["config"] or {}

        # Fetch analytics
        analysis = await self.analytics.get_analytics(pipeline_id)
        evals = analysis["evaluation"]

        suggestions = []

        # 1. Low context precision: precision < 0.6
        if evals["context_precision"] < 0.6:
            current_top_k = config.get("retrieval", {}).get("top_k_after_rerank", 8)
            suggested_top_k = max(1, current_top_k - 3)
            suggestions.append({
                "metric": "context_precision",
                "problem": "too many retrieved chunks or low precision",
                "suggestion": f"reduce top_k_after_rerank from {current_top_k} to {suggested_top_k}"
            })

        # 2. Low context recall: recall < 0.6
        if evals["context_recall"] < 0.6:
            current_dense_k = config.get("retrieval", {}).get("dense_k", 10)
            suggested_dense_k = current_dense_k + 5
            suggestions.append({
                "metric": "context_recall",
                "problem": "not enough relevant chunks retrieved",
                "suggestion": f"increase dense_k from {current_dense_k} to {suggested_dense_k}"
            })

        # 3. Low faithfulness: faithfulness < 0.7
        if evals["faithfulness"] < 0.7:
            current_temp = config.get("generation", {}).get("temperature", 0.7)
            suggested_temp = max(0.0, round(current_temp - 0.2, 2))
            suggestions.append({
                "metric": "faithfulness",
                "problem": "generation contains ungrounded claims",
                "suggestion": f"reduce temperature from {current_temp} to {suggested_temp} and increase reranker quality"
            })

        # 4. Low answer relevance: relevance < 0.7
        if evals["answer_relevance"] < 0.7:
            current_expansion = config.get("retrieval", {}).get("query_expansion", False)
            if not current_expansion:
                suggestions.append({
                    "metric": "answer_relevance",
                    "problem": "answer does not address query intent directly",
                    "suggestion": "enable query expansion"
                })
            else:
                suggestions.append({
                    "metric": "answer_relevance",
                    "problem": "answer relevance remains low despite expansion",
                    "suggestion": "refine query expansion system prompt or model selection"
                })

        # Integrate QualityAnomalyDetector anomalies
        try:
            from monitoring.anomaly_detector import QualityAnomalyDetector
            detector = QualityAnomalyDetector()
            anomalies = await detector.detect_anomalies(pipeline_id)
            suggestions.extend(anomalies)
        except Exception as e:
            logger.error(f"Failed to detect quality anomalies: {e}")

        return suggestions
