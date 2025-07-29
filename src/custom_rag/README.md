# Custom RAG

## Differentiator
> Custom RAG may be appropriate when the out of box RAG (Gen AI Agents) leave room for improvement.
> 
> This module offers an example of a custom RAG pipeline where each part may be customized to suit the needs of the business use case.
> 
> The [Usage Instructions](#usage-instructions) will walk you through how to obtain a response to a query for information from a PDF document, which may be sourced from Object Storage or locally.

## Key Features

* `OCI Gen AI Llama3.3 70b Instruct` as LLM
* `OCI Gen AI Cohere Embed English v3.0` as Embed model
* `ChromaDB` local database (uses sqlite3 Python package) for Vector DB
* `PDF document` sourced from either Object Storage or local file
* `RecursiveCharacterTextSplitter` from `Langchain` to split and chunk document content
* `Langchain` to chain the inputs and outputs into a RAG pipeline

## Functional Notes

* The [Usage Instructions](#usage-instructions) assume that Oracle Linux is used
* Output is printed to the CLI console
* Query is hardcoded in [main.py](./main.py)
* Environment variables are sourced from file: `config/.env`
* OCI Config file variables are sourced from file: `~/.oci/config`
* Document content outside of the provided examples may not be responded to effectively due to limitations of a single API call to Embedding Model, Vector DB, or LLM. Multiple API calls may have to be used to respond to more content.

## Usage Instructions

1. Clone this repository and enter the cloned directory:
    ```
    git clone https://github.com/aojah1/adk_projects.git
    cd adk_projects
    ```
2. Ensure that `Python3.13` is installed on your machine. For `Python3.13` installation details, see repo level [readme.md](../../readme.md).
3. Ensure that your `Python3.13` installation includes a sufficient `sqlite3` version, such as `3.41.2`. You can check `sqlite3` version using the following command:
    ```
    /usr/local/bin/python3.13 -c "import sqlite3; print(sqlite3.sqlite_version)"
    
    # Should return a sufficient version number, such as 3.41.2
    ```
4. Create a virtual environment using `Python3.13`:
    ```
    # Create a virtual environment. Replace the placeholder value <VIRTUAL_ENVIRONMENT_FOLDER_NAME> with a name of your choice.
    python3.13 -m venv <VIRTUAL_ENVIRONMENT_FOLDER_NAME>
    ```
5. Ensure that your virtual environment is activated. You can activate it using the following command. Replace the placeholder `<PATH_TO_VIRTUAL_ENVIRONMENT>` path with your own path.
    ```
    source <PATH_TO_VIRTUAL_ENVIRONMENT>/bin/activate
    ```
6. Install required dependency packages:
    ```
    pip install -r requirements.txt
    ```
7. Your OCI config file must be located at `~/.oci/config`. Ensure that this file is properly configured by following this OCI documentation: https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm
8. Edit environment variables as desired in a new `config/.env` file in `OCI Custom RAG Configuration` section, using `config/sample_.env` as a reference. A sample PDF source document is included at `local_files/finance_data.pdf`. You may choose to source this file from its current location, or upload this PDF document to an Object Storage bucket and source from that location.
    ```
    # Copy the config/sample_.env file to a new file called config/.env
    cp config/sample_.env config/.env
    
    # Edit config/.env and update values
    vi config/.env
    
    # When finished editing config/.env, save and quit the file using :wq
    ```
9. Run the Custom RAG module:
    ```
    python3.13 src/custom_rag/main.py
    ```
10. Observe the response returned from the Custom RAG module, and note whether the returned information is accurate to the source content in the PDF document.