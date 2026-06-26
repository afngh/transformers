import argparse
from .load import dynamo

def main():
    parser = argparse.ArgumentParser(description="Dynamo Text Generation API")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("-l", "--length", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("-k", "--top_k", type=int, default=None, help="Top-k sampling threshold")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling threshold")
    parser.add_argument("-s", "--stream", action="store_true",default=True, help="Stream response token by token")

    args = parser.parse_args()

    client = dynamo()
    client.Client()

    response = client.create(
        input=args.prompt,
        max_tokens=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stream=args.stream
    )

    if args.stream:
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
    else:
        print(response)

if __name__ == '__main__':
    main()