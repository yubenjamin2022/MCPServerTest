import ast, inspect, textwrap
from transformers import AutoModel, AutoTokenizer
import torch  
import torch.nn.functional as F  
import numpy as np

class NaturalLanguageParser():
    """

    Given natural language descriptions of functions, seperate and encode them

    Assumption: file is structured like a JSON file, with each entry being

    Didn't include guardrails (i.e. checking if file is a Python file, etc.), very barebones

    """
    
    def __init__(self, function_dict, model_id):
        self.function_dict = function_dict
        self.function_names = self.get_function_names()
        self.functions = [self.structure_descriptions(fun) for fun in self.function_dict]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()  
        self.embeddings = np.array([self.embed_code(fun).reshape(-1).cpu().detach().numpy() for fun in self.functions])
    
    def get_function_names(self):
        """

        Given the Python file's contents, return all of the relevant function names.

        Assume no async functions for now. 

        """

        return [function_desc['name'] for function_desc in self.function_dict]
    
    def structure_descriptions(self, function):
        """

        Given the Python file's descriptions (structured as a dictionary), 
        format it such that it contains all information in a single string

        """
        
        function_name = f"Function name: {function['name']}"
        function_description = f"Description: {function['description']}"
        required_params = "Required Parameters: \n" + '\n'.join([f"{param['name']} ({param['type']}) \n Description: {param['description']}"] 
                                    for param in function['required_parameters'])
        optional_params = "Optional Parameters: \n" + '\n'.join([f"{param['name']} ({param['type']}) \n Description: {param['description']} \n Default value: {param['default']}"] 
                                    for param in function['optional_parameters'])

        return '\n'.join([function_name, function_description, required_params, optional_params])
    
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