from function_parser import FunctionParser
from natural_language_parser import NaturalLanguageParser
import matplotlib.pyplot as plt
import numpy as np

from bio_tools.microbiology_descriptions import description as microbiology_description
from bio_tools.molecularbiology_descriptions import description as molecularbiology_description

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

    parser_micro = FunctionParser('experiments\bio_tools\microbiology.py', model_id)
    parser_molecular = FunctionParser('experiments\bio_tools\molecularbiology.py', model_id)

    parsers = [parser_micro, parser_molecular]

    for i, parser_1 in enumerate(parsers):
        for j, parser_2 in enumerate(parsers):
            similarity_scores = np.array(
                                        [[cosine_sim(parser_1.embeddings[a], parser_2.embeddings[b]) 
                                          for a in range(len(parser_1.embeddings))] 
                                         for b in range(len(parser_2.embeddings))]
                                        )
            
            fig, ax = plt.subplots()
            im = ax.imshow(similarity_scores)

            # Show all ticks and label them with the respective list entries
            ax.set_xticks(range(len(parser_1.function_names)), labels=parser_1.function_names,
                        rotation=45, ha="right", rotation_mode="anchor")
            ax.set_yticks(range(len(parser_2.function_names)), labels=parser_2.function_names)

            # Loop over data dimensions and create text annotations.
            for c in range(len(parser_2.function_names)):
                for d in range(len(parser_1.function_names)):
                    text = ax.text(d, c, similarity_scores[c, d],
                                ha="center", va="center", color="w")

            ax.set_title(f"Cosine similarity between functions of Class {i} and Class {j}")
            fig.tight_layout()
            plt.savefig(f'experiments/figures/Class_{i}_{j}_function_similarity.png')

    class_similarity_scores = np.zeros((3, 3))

    for i, parser_1 in enumerate(parsers):
        for j, parser_2 in enumerate(parsers):
            class_similarity_scores[i, j] = cosine_sim(parser_1.class_embeddings, parser_2.class_embeddings)

    fig, ax = plt.subplots()
    im = ax.imshow(class_similarity_scores)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(parsers)), labels=range(len(parsers)),
                ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(parsers)), labels=range(len(parsers)))

    # Loop over data dimensions and create text annotations.
    for c in range(len(parsers)):
        for d in range(len(parsers)):
            text = ax.text(d, c, similarity_scores[c, d],
                        ha="center", va="center", color="w")

    ax.set_title(f"Cosine similarity between classes 1, 2, and 3")
    fig.tight_layout()
    plt.savefig(f'experiments/figures/Bio_class_similarity.png')

    #############################################################
    # Embeddings pf only natural language descriptions of functions         
    #############################################################

    NL_parser_micro = NaturalLanguageParser(microbiology_description, model_id)
    NL_parser_molecular = NaturalLanguageParser(molecularbiology_description, model_id)

    NL_parsers = [NL_parser_micro, NL_parser_molecular]

    for i, parser_1 in enumerate(NL_parsers):
        for j, parser_2 in enumerate(NL_parsers):
            similarity_scores = np.array(
                                        [[cosine_sim(parser_1.embeddings[a], parser_2.embeddings[b]) 
                                          for a in range(len(parser_1.embeddings))] 
                                         for b in range(len(parser_2.embeddings))]
                                        )
            
            fig, ax = plt.subplots()
            im = ax.imshow(similarity_scores)

            # Show all ticks and label them with the respective list entries
            ax.set_xticks(range(len(parser_1.function_names)), labels=parser_1.function_names,
                        rotation=45, ha="right", rotation_mode="anchor")
            ax.set_yticks(range(len(parser_2.function_names)), labels=parser_2.function_names)

            # Loop over data dimensions and create text annotations.
            for c in range(len(parser_2.function_names)):
                for d in range(len(parser_1.function_names)):
                    text = ax.text(d, c, similarity_scores[c, d],
                                ha="center", va="center", color="w")

            ax.set_title(f"Cosine similarity between functions of Class {i} and Class {j}")
            fig.tight_layout()
            plt.savefig(f'experiments/figures/Class_{i}_{j}_function_similarity.png')

    class_similarity_scores = np.zeros((3, 3))

    for i, parser_1 in enumerate(parsers):
        for j, parser_2 in enumerate(parsers):
            class_similarity_scores[i, j] = cosine_sim(parser_1.class_embeddings, parser_2.class_embeddings)

    fig, ax = plt.subplots()
    im = ax.imshow(class_similarity_scores)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(parsers)), labels=range(len(parsers)),
                ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(parsers)), labels=range(len(parsers)))

    # Loop over data dimensions and create text annotations.
    for c in range(len(parsers)):
        for d in range(len(parsers)):
            text = ax.text(d, c, similarity_scores[c, d],
                        ha="center", va="center", color="w")

    ax.set_title(f"Cosine similarity between classes 1, 2, and 3")
    fig.tight_layout()
    plt.savefig(f'experiments/figures/Bio_class_similarity.png')