from function_parser import FunctionParser
import matplotlib.pyplot as plt
import numpy as np

def cosine_sim(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a @ b.T).item()

if __name__ == "__main__":
    # model_id = "BAAI/bge-m3"
    model_id = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

    # initialize function parsers for each of the classes & models

    parser_linalg = FunctionParser('experiments\classes\lin_alg_tool.py', model_id)
    parser_mat = FunctionParser('experiments\classes\matrix_analysis_tool.py', model_id)
    parser_deriv = FunctionParser('experiments\classes\derivative_approx.py', model_id)

    parsers = [parser_linalg, parser_mat, parser_deriv]

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
    plt.savefig(f'experiments/figures/Class_similarity.png')