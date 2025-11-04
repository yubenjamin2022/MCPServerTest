from function_parser import FunctionParser
from natural_language_parser import NaturalLanguageParser
import matplotlib.pyplot as plt
import numpy as np

from bio_tools.cell_biology_descriptions import description as cell_biology_description
from bio_tools.biochemistry_descriptions import description as biochemistry_description

def cosine_sim(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a @ b.T).item()

if __name__ == "__main__":
    
    # model_id = "BAAI/bge-m3"
    model_id = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

    # initialize function parsers for each of the classes & models

    #############################################################
    # Embeddings of entire functions/classes                    
    #############################################################

    parser_micro = FunctionParser('experiments/bio_tools/cell_biology.py', model_id, 0)
    parser_molecular = FunctionParser('experiments/bio_tools/biochemistry.py', model_id, 0)

    parsers = [parser_micro, parser_molecular]

    similarity_scores = np.array(
                        [[cosine_sim(parser_micro.embeddings[a], parser_molecular.embeddings[b]) 
                            for a in range(len(parser_micro.embeddings))] 
                            for b in range(len(parser_molecular.embeddings))]
                        )

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(similarity_scores, cmap='Blues')

    ax.set_xticks(range(len(parser_micro.function_names)), labels=parser_micro.function_names,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(parser_molecular.function_names)), labels=parser_molecular.function_names)

    for c in range(len(parser_molecular.embeddings)):     
        for d in range(len(parser_micro.embeddings)):  
            text = ax.text(d, c, np.round(similarity_scores[c, d], 2),
                        ha="center", va="center", color="black")

    ax.set_title(f"Cosine similarity between Microbiology and Molecular Biology Functions")
    fig.tight_layout()
    fig.savefig(f'experiments/figures/Bio_function_similarity.png')

    #############################################################
    # Embeddings of only natural language descriptions of functions         
    #############################################################

    NL_parser_micro = NaturalLanguageParser(cell_biology_description, model_id)
    NL_parser_molecular = NaturalLanguageParser(biochemistry_description, model_id)

    similarity_scores = np.array(
            [[cosine_sim(NL_parser_micro.embeddings[a], NL_parser_molecular.embeddings[b]) 
                for a in range(len(NL_parser_micro.embeddings))] 
                for b in range(len(NL_parser_molecular.embeddings))]
        )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(similarity_scores, cmap='Blues')

    ax.set_xticks(range(len(NL_parser_micro.function_names)), labels=NL_parser_micro.function_names,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(NL_parser_molecular.function_names)), labels=NL_parser_molecular.function_names)

    for c in range(len(NL_parser_molecular.embeddings)):     
        for d in range(len(NL_parser_micro.embeddings)):  
            text = ax.text(d, c, np.round(similarity_scores[c, d], 2),
                        ha="center", va="center", color="black")

    ax.set_title(f"NL Cosine similarity between Microbiology and Molecular Biology Functions")
    fig.tight_layout()
    fig.savefig(f'experiments/figures/NL_function_similarity.png')