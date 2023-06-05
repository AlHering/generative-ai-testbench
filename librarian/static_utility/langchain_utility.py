# -*- coding: utf-8 -*-
"""
****************************************************
*           langchain_testbench:librarian                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import List
from chromadb.config import Settings
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from hashing_utility import hash_text_with_sha256


"""
ChromaDB
"""


def get_or_create_chromadb(chroma_db_settings: Settings, embedding_model: BaseModel) -> Chroma:
    """
    Function for acquiring ChromaDB instance.
    :param chroma_db_settings: ChromaDB Settings object.
    :param embedding_model: Embedding model.
    """
    return Chroma(persist_directory=chroma_db_settings.persist_directory, embedding_function=embedding_model, client_settings=chroma_db_settings)


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
