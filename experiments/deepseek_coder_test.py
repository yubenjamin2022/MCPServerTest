# Example for DeepSeek-Coder embedding + cosine similarity  
from transformers import AutoTokenizer, AutoModel  
import torch  
import torch.nn.functional as F  

# load model & tokenizer  
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)  
model = AutoModel.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)  
model.eval()  

def embed_code_ds(code_str):  
    inputs = tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=2048)  
    with torch.no_grad():  
        outputs = model(**inputs)  
        # assume last_hidden_state first token  
        cls_emb = outputs.last_hidden_state[:,0]  
        cls_emb = F.normalize(cls_emb, p=2, dim=1)  
    return cls_emb  

def cosine_sim(a, b):  
    return (a @ b.T).item()  

func1 = """def count_lines(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)
"""  
func2 = """def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)"""  

emb1_ds = embed_code_ds(func1)  
emb2_ds = embed_code_ds(func2)  

print("DeepSeek-Coder similarity:", cosine_sim(emb1_ds, emb2_ds))
