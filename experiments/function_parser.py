import ast, inspect, textwrap
from transformers import AutoModel, AutoTokenizer
import torch  
import torch.nn.functional as F  
import numpy as np

class FunctionParser():
    """

    Given a Python tool (examples in classes), seperate into functions and embed it 

    Assumption: the files are python classes, which are valid with comments

    Didn't include guardrails (i.e. checking if file is a Python file, etc.), very barebones

    """
    
    def __init__(self, filepath, model_id, cutoff = 2):
        self.filepath = filepath
        self.content = self.read_file(filepath)
        self.function_names = self.get_function_names()
        self.functions = self.extract_functions_from_class()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()  
        self.embeddings = np.array([self.embed_code(fun).reshape(-1).cpu().detach().numpy() for fun in self.functions])
        self.class_embeddings = self.embed_code(self.content)
        self.cutoff = cutoff

    def read_file(self, filepath) -> str:
        """
        
        Reads the file path and returns the content of the file. 

        Arguments:

        filepath: filepath of the Python file

        Returns: (str) the content of the Python file

        """

        with open(filepath, 'r') as file:
            file_content = file.read()
        
        return file_content
    
    def get_function_names(self):
        """

        Given the Python file's contents, return all of the relevant function names.

        Assume no async functions for now. 

        """

        function_names = []
        with open(self.filepath, 'r') as file:
            tree = ast.parse(file.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
        return function_names
    
    def extract_functions_from_class(self):
        """

        Given the Python file's contents, return all the (relevant) functions (i.e. not the __init__ or the run functions)

        Assumption: the Python file is a class with at least 3 functions, __init__, run, and at least one tool

        """
        tree = ast.parse(self.content)
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for body_item in node.body:
                    if isinstance(body_item, ast.FunctionDef):
                        # Get exact function source by slicing from node.lineno
                        func_source = textwrap.dedent(
                            "\n".join(self.content.splitlines()[body_item.lineno - 1 : body_item.end_lineno])
                        )
                        functions.append(func_source)
        return functions[self.cutoff:]
    
    def embed_code(self, code_str):
        """

        Given a function, embed it using the model.

        Arguments:
        
        code_str (str): code for Python function

        Returns:

        mean_emb (torch.Tensor): embedding for the function

        """
        inputs = self.tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(**inputs)
            mean_emb = outputs.last_hidden_state.mean(dim=1)
            mean_emb = F.normalize(mean_emb, p=2, dim=1)
        return mean_emb