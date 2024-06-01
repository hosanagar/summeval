import openai
import json
import argparse
import tqdm
import time
import concurrent.futures
import tiktoken


def process_instance(instance, prompt, model):
    source = instance["source"]
    system_output = instance["system_output"]
    cur_prompt = prompt.replace("{{Document}}", source).replace(
        "{{Summary}}", system_output
    )
    instance["prompt"] = cur_prompt
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": cur_prompt}],
            temperature=1,
            max_tokens=args.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            n=20,
        )
        time.sleep(0.5)  # Sleep to respect potential rate limits
        instance["all_responses"] = [
            choice.message.content for choice in response.choices
        ]

        return instance

    except Exception as e:
        print(f"Error: {e}")
        return None


def main(args):
    openai.api_key = args.key
    summeval = json.load(open(args.summeval_fp))
    prompt = open(args.prompt_fp).read()

    summeval = #summeval[:800]

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        new_json = list(
            tqdm.tqdm(
                executor.map(
                    process_instance,
                    summeval,
                    [prompt] * len(summeval),
                    [args.model] * len(summeval),
                ),
                total=len(summeval),
            )
        )

    with open(args.save_fp, "w") as f:
        json.dump([instance for instance in new_json if instance], f, indent=4)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt_fp", type=str, required=True)
    argparser.add_argument("--save_fp", type=str, required=True)
    argparser.add_argument(
        "--summeval_fp", type=str, default="data/summeval/summeval.json"
    )
    argparser.add_argument("--max_tokens", type=int, default="200")
    argparser.add_argument(
        "--key", type=str, default=""
    )
    argparser.add_argument("--model", type=str)  # default="gpt-3.5-turbo-0125")
    args = argparser.parse_args()

    main(args)
