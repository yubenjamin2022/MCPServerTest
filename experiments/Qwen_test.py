from transformers import AutoModel, AutoTokenizer
import torch  
import torch.nn.functional as F  

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

#############################################################
# TASK 1: Two simple and same functions                     
#############################################################
func1_task1 = """def add(a, b):\n    return a + b"""  
func2_task1 = """def sum_two(x, y):\n    result = x + y\n    return result"""  

emb1_task1 = embed_code(func1_task1)  
emb2_task1 = embed_code(func2_task1)  

print("Qwen2 similarity for two simple, same functions:", cosine_sim(emb1_task1, emb2_task1))

#############################################################
# TASK 2: Two simple and different functions                     
############################################################# 
func1_task2 = """def count_lines(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)
"""  
func2_task2 = """def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)"""  

emb1_task2 = embed_code(func1_task2)  
emb2_task2 = embed_code(func2_task2)  

print("Qwen2 similarity for 2 small, different Python functions:", cosine_sim(emb1_task2, emb2_task2))

#############################################################
# TASK 3: Two longer and similar functions                     
############################################################# 

func1_task3 = """def clean_text_v1(text):
    import re
    stopwords = {'the', 'is', 'and', 'to', 'of', 'a', 'in'}
    
    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    
    # Remove stopwords
    words = [w for w in text.split() if w not in stopwords]
    cleaned = ' '.join(words)
    
    # Normalize whitespace again
    cleaned = re.sub(r'\\s+', ' ', cleaned)
    return cleaned
"""

func2_task3 = """def normalize_text_v2(sentence, remove_stopwords=True):
    import re
    stopwords = {'the', 'is', 'and', 'to', 'of', 'a', 'in'}
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9 ]', ' ', sentence)
    tokens = [word for word in sentence.split() if word]
    
    if remove_stopwords:
        tokens = [w for w in tokens if w not in stopwords]
    
    result = ' '.join(tokens)
    result = re.sub(r'\\s+', ' ', result)
    return result
"""
emb1_task3 = embed_code(func1_task3)  
emb2_task3 = embed_code(func2_task3)  

print("Qwen2 similarity for 2 more complex, same Python functions:", cosine_sim(emb1_task3, emb2_task3))


#############################################################
# TASK 4: Two longer and similar functions                     
############################################################# 

func1_task4 = """def compute_average_per_category(data):
    from collections import defaultdict
    
    stats = defaultdict(list)
    for category, value in data:
        try:
            val = float(value)
            stats[category].append(val)
        except ValueError:
            continue  # Skip invalid values
    
    summary = {}
    for cat, vals in stats.items():
        if not vals:
            continue
        summary[cat] = {
            'mean': sum(vals) / len(vals),
            'min': min(vals),
            'max': max(vals),
            'count': len(vals)
        }
    return summary
"""

func2_task4 = """def generate_random_password(length=16, include_symbols=True):
    import random, string
    
    letters = string.ascii_letters
    digits = string.digits
    symbols = string.punctuation if include_symbols else ''
    
    if length < 6:
        raise ValueError("Password length must be at least 6 characters.")
    
    # Ensure at least one of each type
    base = [
        random.choice(letters),
        random.choice(digits),
        random.choice(symbols or letters)
    ]
    
    remaining = [random.choice(letters + digits + symbols) for _ in range(length - len(base))]
    password_chars = base + remaining
    random.shuffle(password_chars)
    
    password = ''.join(password_chars)
    return password
"""
emb1_task4 = embed_code(func1_task4)  
emb2_task4 = embed_code(func2_task4)  

print("Qwen2 similarity for 2 more complex, different Python functions:", cosine_sim(emb1_task4, emb2_task4))

#############################################################
# TASK 5: Two longer and similar functions (bad variable names)                    
############################################################# 

func1_task5 = """def do_stuff_v1(x):
    import re
    zzz = {'at', 'on', 'for', 'it', 'an', 'the'}
    
    # lowercase
    x = x.lower()
    
    # random cleanup
    x = re.sub(r'[^a-z0-9\\s]', ' ', x)
    x = re.sub(r'\\s+', ' ', x).strip()
    
    # some filtering
    y = [p for p in x.split() if p not in zzz]
    omg = ' '.join(y)
    
    # one more time
    omg = re.sub(r'\\s+', ' ', omg)
    return omg
"""

func2_task5 = """def do_other_thing_v2(q, hmm=True):
    import re
    zzz = {'at', 'on', 'for', 'it', 'an', 'the'}
    
    q = q.lower().strip()
    q = re.sub(r'[^a-z0-9 ]', ' ', q)
    w = [t for t in q.split() if t]
    
    if hmm:
        w = [t for t in w if t not in zzz]
    
    wow = ' '.join(w)
    wow = re.sub(r'\\s+', ' ', wow)
    return wow
"""

emb1_task5 = embed_code(func1_task5)
emb2_task5 = embed_code(func2_task5)

print("Qwen2 similarity for 2 similar messy functions:", cosine_sim(emb1_task5, emb2_task5))

#############################################################
# TASK 6: Two longer and different functions (bad variable names)                    
############################################################# 

func1_task6 = """def x9(data):
    from collections import defaultdict
    stuff = defaultdict(list)
    
    for a, b in data:
        try:
            v = float(b)
            stuff[a].append(v)
        except:
            continue
    
    lol = {}
    for k, u in stuff.items():
        if not u:
            continue
        lol[k] = {
            'avg': sum(u) / len(u),
            'lo': min(u),
            'hi': max(u),
            'n': len(u)
        }
    return lol
"""

func2_task6 = """def make_id(thing='X', num=9):
    import random, string
    box = string.ascii_uppercase + string.digits
    
    if num < 4:
        raise ValueError("too short")
    
    tmp = ''.join(random.choice(box) for _ in range(num))
    return f"{thing}-{tmp}"
"""

emb1_task6 = embed_code(func1_task6)
emb2_task6 = embed_code(func2_task6)

print("Qwen2 similarity for 2 very different messy functions:", cosine_sim(emb1_task6, emb2_task6))

