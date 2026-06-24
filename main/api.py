from .load import dynamo

client = dynamo()

client.Client()

response = client.create(
    input="The weather today is",
    max_tokens=50,
    temperature=0.8
)

print(response)