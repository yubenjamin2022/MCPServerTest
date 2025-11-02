from transformers import AutoModel, AutoTokenizer
import torch  
import torch.nn.functional as F  

# model_id = "BAAI/bge-m3"
model_id = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()  

def embed_code(code_str):
    inputs = tokenizer(code_str, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)
        mean_emb = outputs.last_hidden_state.mean(dim=1)
        mean_emb = F.normalize(mean_emb, p=2, dim=1)
    return mean_emb

def cosine_sim(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a @ b.T).item()

file_paths = ['experiments\classes\lin_alg_tool.py', 'experiments\classes\matrix_analysis_tool.py', 'experiments\classes\derivative_approx.py'] 
file_content = []

for file_path in file_paths:
    with open(file_path, 'r') as file:
        file_content.append(file.read())


emb1_task1 = embed_code(file_content[0])  
emb2_task1 = embed_code(file_content[1])  
emb3_task1 = embed_code(file_content[2])  

print("Model similarity for two similar classes:", cosine_sim(emb1_task1, emb2_task1))

print("Model similarity for two different classes (class 2 and 3):", cosine_sim(emb2_task1, emb3_task1))

print("Model similarity for two different classes (class 1 and 3):", cosine_sim(emb1_task1, emb3_task1))
