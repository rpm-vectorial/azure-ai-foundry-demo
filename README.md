# Azure AI Foundry Demo

This project demonstrates how to use Azure AI services to build a chatbot that retrieves product documents from a search index and generates responses based on user queries.

## Project Structure
azure-ai-foundry-demo/ ├── pycache/ ├── assets/ │ ├── grounded_chat_call.prompty │ ├── intent_mapping.prompty │ ├── products.csv ├── chat_with_products.py ├── config.py ├── create_search_index.py ├── get_product_documents.py ├── requirements.txt ├── test.py

## Files

- `chat_with_products.py`: Contains the `chat_with_products` function that facilitates a chat interaction leveraging product information retrieved from a search index.
- `config.py`: Configuration settings and environment variable handling.
- `create_search_index.py`: Script to create a search index and upload product data from a CSV file.
- `get_product_documents.py`: Contains the `get_product_documents` function that retrieves product documents from the search index.
- `requirements.txt`: Lists the dependencies required for the project.
- `test.py`: Script to test the functionality of the project.

## Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/azure-ai-foundry-demo.git
    cd azure-ai-foundry-demo
    ```

2. **Install Dependencies:**

    Ensure you have Python 3.10 or higher installed. Install required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables:**

    Create a `.env` file in the `azure-ai-foundry-demo` directory with the necessary configuration:

    ```env
    AIPROJECT_CONNECTION_STRING=your-connection-string
    AISEARCH_INDEX_NAME=your-index-name
    EMBEDDINGS_MODEL=your-embeddings-model
    INTENT_MAPPING_MODEL=your-intent-mapping-model
    CHAT_MODEL=your-chat-model
    ```

## Usage

### Running the Chatbot

To run the chatbot, execute the `chat_with_products.py` script:

```bash
python [chat_with_products.py](http://_vscodecontentref_/11) --query "I need a new tent for 4 people, what would you recommend?"
