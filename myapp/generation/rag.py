import os
from typing import Dict, List, Optional, Tuple

from groq import Groq
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
        You are an expert product advisor who evaluates retrieved e-commerce products with critical thinking and careful comparison in order to choose the single best option for the user.

        ## Instructions:
        1. Identify the one product that best fulfills the user's request by interpreting how each attribute matters in practice rather than simply restating specifications. Look for meaningful advantages, tradeoffs, and alignment with the user's real intent.
        2. Present the recommendation clearly in this format:
        - Best Product: [Product PID] [Product Name]
        - Why: [Provide a thoughtful explanation of why this product is the best fit. Go beyond listing features. Connect attributes to real benefits, evaluate comparative strengths, highlight value for money, explain practical implications, and show reasoning rather than description.]
        3. If another product could reasonably suit the user, mention it briefly as an alternative with a short explanation of what use case or preference it satisfies.
        4. If none of the products meaningfully satisfy the user's request, return ONLY this exact phrase:
        "There are no good products that fit the request based on the retrieved results."

        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ...
        - Why: ...
        - Alternative (optional): ...

    """

    def __init__(self) -> None:
        self.provider_preference = os.getenv("LLM_PROVIDER", "groq").lower()
        self.clients = self._initialize_clients()

    def _initialize_clients(self) -> Dict[str, Dict[str, object]]:
        clients: Dict[str, Dict[str, object]] = {}

        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            clients["groq"] = {
                "client": Groq(api_key=groq_key),
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            }

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and OpenAI is not None:
            clients["openai"] = {
                "client": OpenAI(api_key=openai_key),
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            }

        return clients

    def _select_client(self) -> Tuple[Optional[str], Optional[object], Optional[str]]:
        # Build preference order: requested provider first, then the rest
        preference: List[str] = []
        preferred = self.provider_preference
        if preferred in self.clients:
            preference.append(preferred)
        for provider in ("groq", "openai"):
            if provider not in preference and provider in self.clients:
                preference.append(provider)

        for provider in preference:
            config = self.clients.get(provider)
            if config:
                return provider, config["client"], config["model"]  # type: ignore[index]

        return None, None, None

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> Dict[str, Optional[str]]:
        """
        Generate a response using the retrieved search results.
        Returns a dictionary with the generated suggestion and metadata.
        """
        default_response = {
            "text": "RAG is not available. Check your credentials (.env file) or account limits.",
            "provider": None,
            "model": None,
        }

        provider, client, model_name = self._select_client()
        if not client:
            return default_response

        try:
            formatted_results = self._format_results(retrieved_results, top_N)

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results,
                user_query=user_query
            )

            chat_completion = client.chat.completions.create(  # type: ignore[union-attr]
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return {
                "text": generation.strip() if generation else "",
                "provider": provider,
                "model": model_name,
            }
        except Exception as e:  # pragma: no cover - relies on external API
            print(f"Error during RAG generation: {e}")
            return default_response

    @staticmethod
    def _format_results(retrieved_results: list, top_N: int) -> str:
        formatted_results = []
        for res in retrieved_results[:top_N]:
            product_info = f"- PID: {res.pid}\n  Title: {res.title}"
            if res.description:
                desc = res.description[:200] + "..." if len(res.description) > 200 else res.description
                product_info += f"\n  Description: {desc}"
            if res.brand:
                product_info += f"\n  Brand: {res.brand}"
            if res.selling_price:
                product_info += f"\n  Price: ₹{res.selling_price:.2f}"
                if res.actual_price and res.actual_price > res.selling_price:
                    product_info += f" (Original: ₹{res.actual_price:.2f})"
            if res.discount:
                product_info += f"\n  Discount: {res.discount:.0f}% OFF"
            if res.average_rating:
                product_info += f"\n  Rating: {res.average_rating:.1f}/5"
            if res.out_of_stock:
                product_info += "\n  Status: Out of Stock"
            else:
                product_info += "\n  Status: In Stock"
            formatted_results.append(product_info)

        return "\n\n".join(formatted_results)
