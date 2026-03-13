import sys, os
import torch

# 1. Environment Setup
VENV_PATH = "/home/jupyter/.cache/pypoetry/virtualenvs/stat359-su25-t0ZXeGPH-py3.10" 
LIB_PATH = os.path.join(VENV_PATH, "lib/python3.10/site-packages")

if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)
    sys.path.append(os.getcwd())

from instructor.final_project.arithmetic_llm.interactive_solver import InteractiveArithmeticSolver

# 2. Load the Baseline Model (Just to borrow its tokenizer)
model_path = "models/instruction_20260302_162638_395697/best_model.pt"
tokenizer_path = "data/tokenizer"

solver = InteractiveArithmeticSolver(
    model_path=model_path, 
    tokenizer_path=tokenizer_path, 
    device="cpu"
)

tokenizer = solver.tokenizer

# 3. The Inputs We Want to X-Ray
test_strings = [
    ("THE DIVISION TEST", "Evaluate: 10 / 2"),
    ("THE DECIMAL TEST", "Evaluate: 10.5 + 4.2")
]

print("\n" + "="*60)
print("TOKENIZER X-RAY: WHAT THE MODEL ACTUALLY SEES")
print("="*60)

for name, text in test_strings:
    print(f"\n{name}")
    print(f"Human Input:  '{text}'")
    
    # Encode the text into raw numerical IDs
    if hasattr(tokenizer, 'encode'):
        encoded = tokenizer.encode(text)
        token_ids = encoded.ids if hasattr(encoded, 'ids') else encoded
    else:
        token_ids = tokenizer(text)['input_ids']
    
    # Translate the IDs back into text to see what survived
    surviving_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    print(f"Token IDs:    {token_ids}")
    print(f"Model Sees:   {surviving_tokens}")
    print("-" * 40)
