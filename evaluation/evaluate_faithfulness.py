
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

load_dotenv()


async def main():
    # Reuse the Azure OpenAI configuration.
    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-10-21"
        ),
    )

    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    # The evaluator LLM checks the RAG response.
    evaluator_llm = llm_factory(
        deployment,
        client=client,
    )

    # Temporary GPT-5 compatibility workaround
    original_create = evaluator_llm.client.chat.completions.create

    async def compatible_create(*args, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        return await original_create(*args, **kwargs)

    evaluator_llm.client.chat.completions.create = compatible_create

    faithfulness = Faithfulness(llm=evaluator_llm)

    faithfulness = Faithfulness(llm=evaluator_llm)

    # Correctly grounded answer
    result_good = await faithfulness.ascore(
        user_input="How often must passwords be changed?",
        response="Passwords must be changed every 90 days.",
        retrieved_contexts=[
            "Employees must change their passwords every 90 days."
        ],
    )

    print("Correct answer score:", result_good.value)

    # Answer contradicts the retrieved document
    result_bad = await faithfulness.ascore(
        user_input="How often must passwords be changed?",
        response="Passwords must be changed every 30 days.",
        retrieved_contexts=[
            "Employees must change their passwords every 90 days."
        ],
    )

    print("Incorrect answer score:", result_bad.value)


if __name__ == "__main__":
    asyncio.run(main())
