from src.llm_model import LLModel


model = LLModel()
messages = model.generate_prompt("What is the capital of France?")
print(model.generate(messages))
