# 1. This script retrieves product documents and generates a response to a user's question.

# 2. Add code to import the required libraries, create a project client, and configure settings 
import os
from pathlib import Path
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from config import ASSET_PATH, get_logger, enable_telemetry
from get_product_documents import get_product_documents

# initialize logging and tracing objects
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


# create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# create a chat client that can be used for testing
chat = project.inference.get_chat_completions_client()


# 3. Create the chat function that uses the RAG capabilities
from azure.ai.inference.prompts import PromptTemplate
@tracer.start_as_current_span(name="chat_with_products")
### The chat_with_products function is designed to facilitate a chat interaction that leverages product information retrieved from a search index. 
# It starts by importing the PromptTemplate class from the azure.ai.inference.prompts module. 
# The function is decorated with @tracer.start_as_current_span(name="chat_with_products"), which ensures that the function execution is traced for monitoring and debugging purposes.
# The function accepts two parameters: messages, a list of chat messages, and an optional context dictionary. 
# If the context is not provided, it initializes an empty dictionary. The function then calls get_product_documents with the provided messages and context to retrieve relevant product documents from the search index.
# Next, the function prepares for a grounded chat call using the retrieved documents. 
# It loads a prompt template from a file named grounded_chat_call.prompty located in the ASSET_PATH directory. This template is used to create system messages that incorporate the retrieved documents and the current context.
# The function then calls the chat.complete method, passing the model specified in the CHAT_MODEL environment variable, the combined system and user messages, and any additional parameters defined in the prompt template. 
# This method generates a response based on the provided input.
# Finally, the function logs the generated response message and returns a dictionary containing the response message and the updated context. 
# This ensures that the chat interaction is compliant with the expected chat protocol and includes any relevant context for future interactions.'''

def chat_with_products(messages: list, context: dict = None) -> dict:
    if context is None:
        context = {}
    documents = get_product_documents(messages, context)

    # do a grounded chat call using search results
    grounded_chat_prompt = PromptTemplate.from_prompty(Path(ASSET_PATH) / "grounded_chat_call.prompty")
    system_message = grounded_chat_prompt.create_messages(documents=documents, context=context)
    response = chat.complete(
        model=os.environ["CHAT_MODEL"],
        messages=system_message + messages,
        **grounded_chat_prompt.parameters,
    )
    logger.info(f"💬 Response: {response.choices[0].message}")

    # Return a chat protocol compliant response
    return {"message": response.choices[0].message, "context": context}

# 4. Add the code to run the chat function
if __name__ == "__main__":
    import argparse

    # load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Query to use to search product",
        default="I need a new tent for 4 people, what would you recommend?",
    )
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable sending telemetry back to the project",
    )
    args = parser.parse_args()
    if args.enable_telemetry:
        enable_telemetry(True)

    # run chat with products
    response = chat_with_products(messages=[{"role": "user", "content": args.query}])