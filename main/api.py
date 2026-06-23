from .load import dynamo

client = dynamo()

client.Client()

response = client.create(
    input="war",
    max_tokens=100,
    temperature=1.5
)

print(response)