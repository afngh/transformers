import argparse
from .load import dynamo

def main():
    parser = argparse.ArgumentParser(description="Dynamo Text Generation API")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("-l", "--length", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Sampling temperature")

    args = parser.parse_args()

    client = dynamo()
    client.Client()

    response = client.create(
        input=args.prompt,
        max_tokens=args.length,
        temperature=args.temperature
    )

    print(response)

if __name__ == '__main__':
    main()