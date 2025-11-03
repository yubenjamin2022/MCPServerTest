from function_parser import FunctionParser
import matplotlib.pyplot as plt
import numpy as np

def cosine_sim(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a @ b.T).item()

if __name__ == "__main__":
    # initialize all classes & models 
    file_paths = ['experiments\classes\lin_alg_tool.py', 'experiments\classes\matrix_analysis_tool.py', 'experiments\classes\derivative_approx.py'] 
    # model_id = "BAAI/bge-m3"
    model_id = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

    # initialize function parsers for each of the classes & models

    parser_linalg = FunctionParser('experiments\classes\lin_alg_tool.py', model_id)
    parser_mat = FunctionParser('experiments\classes\matrix_analysis_tool.py', model_id)
    parser_deriv = FunctionParser('experiments\classes\derivative_approx.py', model_id)

    parsers = [parser_linalg, parser_mat, parser_deriv]

    for i, parser in enumerate(parsers):