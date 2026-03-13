import torch
from instructor.final_project.arithmetic_llm.interactive_solver import InteractiveArithmeticSolver

# Load the Baseline Model
model_path = "models/instruction_20260302_162638_395697/best_model.pt"
tokenizer_path = "data/tokenizer"

solver = InteractiveArithmeticSolver(
    model_path=model_path, 
    tokenizer_path=tokenizer_path, 
    device="cuda" if torch.cuda.is_available() else "cpu"
)

demo_prompts = [
    ("SCENARIO 1: IN-DISTRIBUTION SUCCESS", "15 + 4"),
    ("SCENARIO 2: ATTENTION OVERLOAD (OOD)", "4035 + 8268"),
    ("SCENARIO 3: TOKENIZER WALL (OOD)", "10 / 2")
]

print("\n" + "="*60)
print("ARITHMETIC LLM: REASONING DEMO")
print("="*60)

for scenario_name, expression in demo_prompts:
    print(f"\n{scenario_name}")
    print(f"User Input: Evaluate: {expression}")
    print("-" * 40)
    print("Model Output:\n")
    
    # THE FIX: We actually capture the returned text and print it!
    try:
        result = solver.solve(expression)
        print(result)
    except Exception as e:
        print(f"[Model Crashed or Returned Error]: {e}")
    
    print("="*60)
