# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:librarian                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import List, Tuple
from langchain.docstore.document import Document
from multiprocessing import Pool
from tqdm import tqdm
from static_utility import langchain_utility


class Librarian(object):
    """
    Class, representing an LLM-based librarian agent to interactively query document libraries.
    """

    def __init__(self, profile: dict) -> None:
        """
        Initiation method.
        :param profile: Profile, configuring a librarian agent.
        """
        self.profile = profile
        self.chroma_db = langchain_utility.get_or_create_chromadb(
            profile["db"])

    def reload_folder(self, folder: str) -> None:
        """
        Method for (re)loading folder contents.
        :param folder: Folder path.
        """
        documents = []
        content_batches = []
        for root, dirs, files in os.walk(folder, topdown=True):
            for file in files:
                _, ext = os.path.splitext(file)
                if any(loader_option.endswith(ext) for loader_option in langchain_utility.DOCUMENT_LOADERS):
                    documents.append((os.path.join(root, file), ext))
        with Pool(processes=os.cpu_count()) as pool:
            with tqdm(total=len(documents), desc='(Re)loading folder contents...', ncols=80) as progress_bar:
                for index, document_contents in enumerate(pool.imap_unordered(self.reload_document, documents)):
                    content_batches.append(document_contents)
                    progress_bar.update(index)
        contents, metadata_entries = self.enrich_content_batches(
            content_batches)
        self.add_contents_to_db(contents, metadata_entries)

    def reload_document(self, document_path: str, extension: str = None) -> List[Document]:
        """
        Method for (re)loading document content.
        :param document_path: Document path.
        :param extension: Extension of file.
            Defaults to None in which case extension is derived from document path.
        """
        return langchain_utility.DOCUMENT_LOADERS[extension if extension is not None else os.path.splitext(document_path)[1]](document_path).load()

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
                                         "part": part_index})
        return contents, metadata_entries

    def add_contents_to_db(self, document_contents: List[str], document_metadata: List[str]) -> None:
        """
        Method for adding document contents to DB.
        :param document_contents: Document contents.
        :param document_metadata: Metadata entries.
        """
        langchain_utility.add_documents_to_chromadb(
            self.chroma_db, document_contents, document_metadata)
