from main.generator_config._load_config_and_model import PretrainedHandler

model_path = 'bin/model/model.pt'
config_path = 'bin/data/config.pkl'

handler = PretrainedHandler(model_path, config_path)

model, config = handler.load()

client = handler.client(model, config, require_params=True, temperature=2, max_tokens=100)

print(client.generate_response(input("enter prompt: ")))