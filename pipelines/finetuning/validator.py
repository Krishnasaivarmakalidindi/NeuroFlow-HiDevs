import re
import logging
from typing import Tuple, Dict, Any

try:
    import tiktoken
except ImportError:
    tiktoken = None

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class TrainingDataValidator:
    def __init__(self):
        self.email_regex = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        self.phone_regex = re.compile(r"(\+\d{1,3})?[\s-]?\d{10}")
        self.citation_regex = re.compile(r"\[Source \d+\]")
        
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        except Exception:
            self.encoding = None

    def _count_tokens(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        # Fallback approximation
        return len(text.split())

    def validate_pair(self, pair: Dict[str, Any]) -> Tuple[bool, str]:
        with tracer.start_as_current_span("finetune.validate") as span:
            span.set_attribute("pair_id", pair.get("id", ""))
            
            # 1. Assistant message length: < 50 tokens or > 2000 tokens -> reject
            assistant_msg = pair.get("assistant_message", "")
            tokens = self._count_tokens(assistant_msg)
            if tokens < 50 or tokens > 2000:
                span.set_attribute("reject_reason", "assistant_length")
                return False, f"assistant_length: {tokens} tokens"
                
            # 2. Missing citation: must contain [Source N] -> reject if missing
            if not self.citation_regex.search(assistant_msg):
                span.set_attribute("reject_reason", "missing_citations")
                return False, "missing_citations"
                
            # 3. Faithfulness: require faithfulness > 0.8 -> reject if <= 0.8 or None
            faithfulness = pair.get("faithfulness")
            if faithfulness is None or faithfulness <= 0.8:
                span.set_attribute("reject_reason", "low_faithfulness")
                return False, f"low_faithfulness: {faithfulness}"
                
            # 4. PII: email or phone number in user or assistant message -> reject
            user_msg = pair.get("user_message", "")
            system_msg = pair.get("system_prompt", "")
            
            for field_name, text in [("user_message", user_msg), ("assistant_message", assistant_msg), ("system_prompt", system_msg)]:
                if self.email_regex.search(text) or self.phone_regex.search(text):
                    span.set_attribute("reject_reason", f"pii_detected_in_{field_name}")
                    return False, f"pii_detected_in_{field_name}"
            
            return True, "valid"
