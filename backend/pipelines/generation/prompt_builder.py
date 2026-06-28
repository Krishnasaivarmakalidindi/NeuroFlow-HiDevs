class PromptBuilder:
    @staticmethod
    def build(query: str, context: str, query_type: str) -> str:
        base_prompt = (
            "You are a precise research assistant.\n\n"
            "Answer the user's question using ONLY the provided context.\n\n"
            "If the context does not contain enough information,\n"
            "say so explicitly.\n\n"
            "For every factual claim,\n"
            "include citations in the format:\n\n"
            "[Source N]\n\n"
            "Never introduce information not present\n"
            "in the provided context."
        )

        query_instructions = ""
        if query_type == "factual":
            query_instructions = (
                "Provide a direct concise answer.\n"
                "If multiple sources agree,\n"
                "cite all of them."
            )
        elif query_type == "analytical":
            query_instructions = (
                "Analyze and synthesize.\n"
                "Identify agreements\n"
                "and contradictions."
            )
        elif query_type == "comparative":
            query_instructions = (
                "Provide structured comparison.\n"
                "Use tables if useful."
            )
        elif query_type == "procedural":
            query_instructions = (
                "Provide numbered steps.\n"
                "Every step must contain citations."
            )

        # For analytical and comparative, add CoT instructions
        cot_instruction = ""
        if query_type in ("analytical", "comparative"):
            cot_instruction = (
                "\n\nYou must start your response with a <think> block containing your detailed step-by-step reasoning before answering the question. Format:\n"
                "<think>\n"
                "step by step reasoning\n"
                "</think>"
            )

        full_prompt = (
            f"{base_prompt}\n\n"
            f"{query_instructions}"
            f"{cot_instruction}\n\n"
            f"<context>\n\n"
            f"{context}\n\n"
            f"</context>\n\n"
            f"Question:\n"
            f"{query}"
        )
        return full_prompt
