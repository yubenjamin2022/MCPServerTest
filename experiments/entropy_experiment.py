from function_parser import FunctionParser
import quests.entropy as entropy
import numpy as np

if __name__ == "__main__":
    parser = FunctionParser('experiments\classes\lin_alg_tool.py', "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    parser2 = FunctionParser('experiments\classes\matrix_analysis_tool.py', "Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    print(parser.embeddings.shape)
    print(entropy.delta_entropy(parser2.embeddings, parser.embeddings, h = 1))