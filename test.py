import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from config import get_logger

# 1. Initialize a logging object to log messages.
logger = get_logger(__name__)

# 2. Create a project client using environment variables loaded from the .env file.
#    This client will be used to interact with the AI project.
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# 3. Create a vector embeddings client that will be used to generate vector embeddings.
#    Vector embeddings are numerical representations of data.
embeddings = project.inference.get_embeddings_client()

# 4. Use the project client to get the default search connection.
#    This connection includes credentials and is used to interact with Azure AI Search.
search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)

# 5. Create a search index client using the search connection.
#    This client will be used to create and delete search indexes.
index_client = SearchIndexClient(
    endpoint=search_connection.endpoint_url, credential=AzureKeyCredential(key=search_connection.key)
)
print(search_connection)
print(index_client)