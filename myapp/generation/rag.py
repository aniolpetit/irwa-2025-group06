import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## Instructions:
        1. Identify the single best product that matches the user's request.
        2. Present the recommendation clearly in this format:
        - Best Product: [Product PID] [Product Name]
        - Why: [Explain in plain language why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user’s needs.]
        3. If there is another product that could also work, mention it briefly as an alternative.
        4. If no product is a good fit, return ONLY this exact phrase:
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

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> dict:
        """
        Generate a response using the retrieved search results. 
        Returns:
            dict: Contains the generated suggestion and the quality evaluation.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        try:
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # Format the retrieved results for the prompt with comprehensive product information
            formatted_results = []
            for res in retrieved_results[:top_N]:
                product_info = f"- PID: {res.pid}\n  Title: {res.title}"
                if res.description:
                    # Truncate description if too long
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
            
            formatted_results_str = "\n\n".join(formatted_results)

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results_str,
                user_query=user_query
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER
