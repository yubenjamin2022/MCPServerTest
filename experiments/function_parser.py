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
        self.cutoff = cutoff
        self.content = self.read_file(filepath)
        self.function_names = self.get_function_names()
        self.functions = self.extract_functions_from_class()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()  
        self.embeddings = [self.embed_code(fun) for fun in self.functions]
        self.class_embeddings = self.embed_code(self.content)

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

        - Ignores async functions (per assumption)
        - Skips nested functions (inside other functions)
        - Keeps both top-level functions and class methods
        """
        function_names = []

        with open(self.filepath, 'r') as file:
            tree = ast.parse(file.read())

        # Only check top-level nodes, not recursively
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for body_item in node.body:
                    if isinstance(body_item, ast.FunctionDef):
                        function_names.append(body_item.name)

        return function_names[self.cutoff:]
    
    def extract_functions_from_class(self):
        """
        Given the Python file's contents, return all the (relevant) functions 
        (i.e. not the __init__ or the run functions).

        Assumption: the Python file is a class with at least 3 functions: 
        __init__, run, and at least one tool function.
        """
        tree = ast.parse(self.content)
        functions = []

        # Iterate only through top-level statements
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # Only get top-level methods in the class, not nested ones
                for body_item in node.body:
                    if isinstance(body_item, ast.FunctionDef):
                        # Skip __init__ and run
                        if body_item.name in {"__init__", "run"}:
                            continue
                        func_source = textwrap.dedent(
                            "\n".join(
                                self.content.splitlines()[body_item.lineno - 1 : body_item.end_lineno]
                            )
                        )
                        functions.append(func_source)
            elif isinstance(node, ast.FunctionDef):
                # Include top-level functions (if needed)
                if node.name in {"__init__", "run"}:
                    continue
                func_source = textwrap.dedent(
                    "\n".join(
                        self.content.splitlines()[node.lineno - 1 : node.end_lineno]
                    )
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