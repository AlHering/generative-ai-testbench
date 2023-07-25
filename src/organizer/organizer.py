# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:organizer                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import List, Tuple
import pandas as pd
from pandas import Series

from src.librarian.librarian import Librarian


class Organizer(Librarian):
    """
    Class, representing an LLM-based organizer agent to collect, embed, cluster and organize texts or documents.
    """

    def __init__(self, profile: dict) -> None:
        """
        Initiation method.
        :param profile: Profile, configuring a organizer agent. The profile should be a nested dictionary of the form
            'chromadb_settings': ChromaDB Settings.
            'embedding':
                'embedding_model': Embedding model.
                'embedding_function': Embedding function.
            'retrieval': 
                'source_chunks': Source chunks.
        """
        super().__init__(profile)

    # Override
    def enrich_content_batches(self, file_paths: List[str], content_batches: List[list]) -> Tuple[list, list]:
        """
        Method for creating metadata for file contents.
        :param file_paths: File paths.
        :param content_batches: Content batches.
        :return: Metadata-enriched contents.
        """
        contents = []
        metadata_entries = []
        for file_index, document_contents in enumerate(content_batches):
            for part_index, document_part in document_contents:
                contents.append(document_part)
                metadata_entries.append({"file_path": file_paths[file_index],
                                         "part": part_index,
                                         "raw": document_part})
        return contents, metadata_entries

    def run_clustering(self) -> None:
        """
        Method for running clustering.
        """
        embedded_docs = self.vector_db.get(include=["embeddings", "metadatas"])
        embedded_docs_df = pd.DataFrame(
            Series(entry) for entry in embedded_docs["embeddings"])

    def choose_topics(self) -> None:
        """
        Method for choosing topics based on clustered texts or documents.
        """
        pass
