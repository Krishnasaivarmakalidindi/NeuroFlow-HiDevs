import os
import functools
import inspect
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

# Global Tracer Setup
_provider_set = False

def get_tracer():
    global _provider_set
    if not _provider_set:
        try:
            # Try to register OTLP gRPC Span Exporter (bound to Jaeger http://localhost:4317)
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            processor = BatchSpanProcessor(exporter)
        except Exception as e:
            # Fallback to ConsoleSpanExporter
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            
        provider = TracerProvider(resource=Resource.create({"service.name": "neuroflow"}))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _provider_set = True
        
    return trace.get_tracer("neuroflow")


def _set_span_attributes_from_args(span, func, args, kwargs, attributes):
    if attributes:
        for k, v in attributes.items():
            if v is not None:
                span.set_attribute(k, v)
                
    # Bind method arguments dynamically to span attributes if they contain RAG meta fields
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        
        target_keys = {
            "pipeline_id", "document_id", "source_type", "page_count", 
            "chunk_count", "token_count", "latency_ms", "model", 
            "run_id", "query_type", "retrieved_chunks", "reranker", 
            "input_tokens", "output_tokens", "cost_usd", "citations",
            "overall_score", "metric_score", "judge_model"
        }
        
        for k, v in bound.arguments.items():
            if k in target_keys and v is not None:
                span.set_attribute(k, str(v) if not isinstance(v, (int, float, bool)) else v)
    except Exception:
        pass


def trace_async(span_name: str, attributes: dict = None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                _set_span_attributes_from_args(span, func, args, kwargs, attributes)
                try:
                    res = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    
                    # Inspect return value for dynamic metrics/attributes
                    if isinstance(res, dict):
                        for k in ["latency_ms", "chunk_count", "token_count", "input_tokens", "output_tokens", "overall_score"]:
                            if k in res and res[k] is not None:
                                span.set_attribute(k, res[k])
                    return res
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise e
        return wrapper
    return decorator


def trace_sync(span_name: str, attributes: dict = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                _set_span_attributes_from_args(span, func, args, kwargs, attributes)
                try:
                    res = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    
                    if isinstance(res, dict):
                        for k in ["latency_ms", "chunk_count", "token_count", "input_tokens", "output_tokens", "overall_score"]:
                            if k in res and res[k] is not None:
                                span.set_attribute(k, res[k])
                    return res
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise e
        return wrapper
    return decorator
