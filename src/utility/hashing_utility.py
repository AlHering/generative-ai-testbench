# -*- coding: utf-8 -*-
"""
****************************************************
*                      utility                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import hashlib


def hash_file_with_sha256(file_path: str) -> str:
    """
    Function for hashing file with SHA256.
    :param file_path: File path.
    :return: Hash.
    """
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(file_path, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def hash_text_with_sha256(text: str) -> str:
    """
    Function for hashing text with SHA256.
    :param text: Text to hash.
    :return: Hash.
    """
    h = hashlib.sha256()
    h.update(bytes(text, "utf-8"))
    return h.hexdigest()
