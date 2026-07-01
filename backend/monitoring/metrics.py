from prometheus_client import Counter, Histogram, Gauge

# --- COUNTERS ---

# Total queries routed through the RAG system
queries_total = Counter(
    "neuroflow_queries_total",
    "Total queries processed by the system",
    labelnames=["pipeline_id", "status"]
)

# Total files ingested
ingestion_docs_total = Counter(
    "neuroflow_ingestion_docs_total",
    "Total documents processed by ingestion",
    labelnames=["pipeline_id", "source_type", "status"]
)

# Total LLM API calls executed
llm_calls_total = Counter(
    "neuroflow_llm_calls_total",
    "Total raw LLM API calls executed",
    labelnames=["pipeline_id", "provider", "model", "status"]
)

# Circuit breaker status changes/trips
circuit_breaker_trips = Counter(
    "neuroflow_circuit_breaker_trips",
    "Total number of times a circuit breaker has tripped",
    labelnames=["pipeline_id", "provider", "status"]
)


# --- HISTOGRAMS ---

# Retrieval latency
retrieval_latency = Histogram(
    "neuroflow_retrieval_latency_seconds",
    "Time taken to execute document chunk retrieval",
    labelnames=["pipeline_id"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Generation latency
generation_latency = Histogram(
    "neuroflow_generation_latency_seconds",
    "Time taken by LLM to stream/generate RAG response",
    labelnames=["pipeline_id", "model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# LLM call monetary cost
llm_cost = Histogram(
    "neuroflow_llm_cost_usd",
    "API cost per query execution",
    labelnames=["pipeline_id", "provider", "model"],
    buckets=[0.0001, 0.001, 0.01, 0.1, 1.0]
)


# --- GAUGES ---

# RAG Faithfulness score
eval_faithfulness = Gauge(
    "neuroflow_eval_faithfulness",
    "Last evaluated RAG response faithfulness score",
    labelnames=["pipeline_id"]
)

# Overall evaluation quality score
eval_overall = Gauge(
    "neuroflow_eval_overall",
    "Last evaluated RAG overall alignment score",
    labelnames=["pipeline_id"]
)

# Redis queue depth
queue_depth = Gauge(
    "neuroflow_queue_depth",
    "Ingestion job worker queue depth",
    labelnames=["pipeline_id"]
)

# Number of circuit breakers open (currently failing/degraded)
circuit_breakers_open = Gauge(
    "neuroflow_circuit_breakers_open",
    "Number of active circuit breakers currently in OPEN state",
    labelnames=["provider"]
)
