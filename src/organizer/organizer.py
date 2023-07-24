# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:organizer                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
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

    def run_clustering(self) -> None:
        """
        Method for running clustering.
        """
        pass

    def choose_topics(self) -> None:
        """
        Method for choosing topics based on clustered texts or documents.
        """
        pass
