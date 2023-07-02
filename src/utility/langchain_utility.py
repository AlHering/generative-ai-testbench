# -*- coding: utf-8 -*-
"""
****************************************************
*                      utility                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from langchain import document_loaders, document_transformers
from typing import List, Any, Optional
from chromadb.config import Settings
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from src.utility.hashing_utility import hash_text_with_sha256


def get_or_create_vectordb(db_type: str = "chromadb", db_kwargs: Optional[Any] = {}) -> Any:
    """
    Method for getting or creating a vector database.
    :param db_type: DB type.
    :param creation_kwargs: Keywords arguments for DB creation or retrieval.
    :return: Vector DB instance.
    """
    if db_type == "chromadb":
        return get_or_create_chromadb(**db_kwargs)


"""
ChromaDB
"""


def get_or_create_chromadb(chroma_db_settings: Settings, embedding_function: Any) -> Chroma:
    """
    Function for acquiring ChromaDB instance.
    :param chroma_db_settings: ChromaDB Settings object.
    :param embedding_function: Embedding function.
    :return: ChromaDB instance.
    """
    return Chroma(persist_directory=chroma_db_settings.persist_directory, embedding_function=embedding_function, client_settings=chroma_db_settings)


def create_collection(chroma_db: Chroma, collection_name: str, metadata: dict = None) -> None:
    """
    Function for adding documents to ChromaDB.
    :param chroma_db: ChromaDB instance.
    :param collection_name: Name of collection to create.
    :param metadata: Collection metadata.
        Defaults to None.
    """
    if metadata is not None:
        chroma_db.Client().get_or_create_collection(
            name=collection_name, metadata=metadata)
    else:
        chroma_db.Client().get_or_create_collection(name=collection_name)


def add_documents_to_chromadb(chroma_db: Chroma, documents: List[str], docs_metadata: List[str] = None, docs_ids: List[str] = None, collection_name: str = None) -> None:
    """
    Function for adding documents to ChromaDB.
    :param chroma_db: ChromaDB instance.
    :param documents: List of (preprocessed) documents.
    :param docs_metadata: List of metadata entries, corresponding to given documents.
        Defaults to None in which case no metadata is added.
    :param docs_ids: List of ids, corresponding to given documents.
        Defaults to None in which case IDs are derived from hashing documents.
    :param collection_name: Name of collection to add documents to.
        Defaults to None in which case base collection is used.
    """
    collection = chroma_db.Client().get_or_create_collection(
        name=collection_name) if collection_name is not None else chroma_db.get()
    kwargs = {
        "documents": documents,
        "ids": docs_ids if docs_ids is not None else [hash_text_with_sha256(text) for text in documents]
    }
    if docs_metadata is not None:
        kwargs["metadatas"] = docs_metadata
    collection.add(**kwargs)


"""
Data Loaders
"""
DOCUMENT_LOADERS = {
    ".csv": document_loaders.CSVLoader,
    ".doc": document_loaders.UnstructuredWordDocumentLoader,
    ".docx": document_loaders.UnstructuredWordDocumentLoader,
    ".enex": document_loaders.EverNoteLoader,
    ".eml": document_loaders.MyElmLoader,
    ".epub": document_loaders.UnstructuredEPubLoader,
    ".html": document_loaders.UnstructuredHTMLLoader,
    ".md": document_loaders.UnstructuredMarkdownLoader,
    ".odt": document_loaders.UnstructuredODTLoader,
    ".pdf": document_loaders.PyMuPDFLoader,
    ".ppt": document_loaders.UnstructuredPowerPointLoader,
    ".pptx": document_loaders.UnstructuredPowerPointLoader,
    ".txt": document_loaders.TextLoader,
    ".json": document_loaders.JSONLoader
}


API_LOADERS = {
    "arxiv": document_loaders.ArxivLoader,
    "azure_blob": document_loaders.AzureBlobStorageFileLoader,
    "onedrive": document_loaders.OneDriveFileLoader
}
