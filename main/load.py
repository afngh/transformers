from main.generator_config._load_config_and_model import PretrainedHandler
import os
from dotenv import load_dotenv

class dynamo:
    def __init__(self):
        load_dotenv(dotenv_path=".env.example")

        self.model_path = os.getenv("MODEL_PATH")
        self.config_path = os.getenv("CONFIG_PATH")
    
        self.handler = PretrainedHandler(self.model_path, self.config_path)

    def Client(self):
        self.model, self.config = self.handler.load()

    def create(self, input :str, temperature=None, max_tokens=None, stream=False):
        model_api = self.handler.client(self.model, self.config, require_params=True, temperature=temperature, max_tokens=max_tokens)

        if stream:
            return model_api.generate_response(input, stream=True)
        else:
            output = model_api.generate_response(input, stream=False)
            return Response(input_text=input, output_text=output)

class Response:
    def __init__(self, input_text=None, output_text=None):
        self.input_text = input_text
        self.output_text = output_text

    def __str__(self):
        return f"""Response(input_text='{self.input_text}',\noutput_text='{self.output_text}')"""