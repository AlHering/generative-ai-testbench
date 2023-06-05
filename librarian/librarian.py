# -*- coding: utf-8 -*-
"""
****************************************************
*           langchain_testbench:librarian                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""


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
