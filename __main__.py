# -*- coding: utf-8 -*-
"""
****************************************************
*             generative_ai_testbench                
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import guidance
from src.configuration import configuration as cfg

# set the default language model used to execute guidance programs
llama = guidance.llms.Transformers(os.path.join(
    cfg.PATHS.TEXTGENERATION_MODEL_PATH, "eachadea_vicuna-7b-1.1"), device=0)

# define the few shot examples
examples = [
    {'input': 'I wrote about shakespeare',
     'entities': [{'entity': 'I', 'time': 'present'}, {'entity': 'Shakespeare', 'time': '16th century'}],
     'reasoning': 'I can write about Shakespeare because he lived in the past with respect to me.',
     'answer': 'No'},
    {'input': 'Shakespeare wrote about me',
     'entities': [{'entity': 'Shakespeare', 'time': '16th century'}, {'entity': 'I', 'time': 'present'}],
     'reasoning': 'Shakespeare cannot have written about me, because he died before I was born',
     'answer': 'Yes'}
]

# define the guidance program
structure_program = guidance(
    '''Given a sentence tell me whether it contains an anachronism (i.e. whether it could have happened or not based on the time periods associated with the entities).
----

{{~! display the few-shot examples ~}}
{{~#each examples}}
Sentence: {{this.input}}
Entities and dates:{{#each this.entities}}
{{this.entity}}: {{this.time}}{{/each}}
Reasoning: {{this.reasoning}}
Anachronism: {{this.answer}}
---
{{~/each}}

{{~! place the real question at the end }}
Sentence: {{input}}
Entities and dates:
{{gen "entities"}}
Reasoning:{{gen "reasoning"}}
Anachronism:{{#select "answer"}} Yes{{or}} No{{/select}}''')


if __name__ == "__main__":
    # execute the program
    out = structure_program(
        examples=examples,
        input='The T-rex bit my dog',
        llm=llama
    )
    print(out["answer"])
