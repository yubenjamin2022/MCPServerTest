# Example for BGE-M3 embedding + cosine similarity  
from transformers import AutoTokenizer, AutoModel  
import torch  
import torch.nn.functional as F  

# load model & tokenizer  
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)  
model = AutoModel.from_pretrained("BAAI/bge-m3", trust_remote_code=True)  
model.eval()  

def embed_code(code_str):  
    inputs = tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=1024)  
    with torch.no_grad():  
        outputs = model(**inputs)  
        # assume using CLS token embedding  
        cls_emb = outputs.last_hidden_state[:,0]  
        # normalize  
        cls_emb = F.normalize(cls_emb, p=2, dim=1)  
    return cls_emb  

def cosine_sim(a, b):  
    return (a @ b.T).item()  

# Two Python function strings  
func1 = """def count_lines(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)
"""  
func2 = """def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)"""  

emb1 = embed_code(func1)  
emb2 = embed_code(func2)  

print("BGE-M3 similarity:", cosine_sim(emb1, emb2))
