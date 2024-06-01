from anthropic import Anthropic
import json
import argparse
import tqdm
import time
import concurrent.futures

client = Anthropic(api_key="")


def process_instance(instance, prompt, model, n):
    source = instance["source"]
    system_output = instance["system_output"]
    cur_prompt = prompt.replace("{{Document}}", source).replace(
        "{{Summary}}", system_output
    )
    all_responses = []

    for i in range(n):
        try:
            response = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": cur_prompt}],
                temperature=1,
                max_tokens=args.max_tokens,
            )
            all_responses.append(response.content[0].text)
            time.sleep(0.5)  # Respect API rate limits
        except Exception as e:
            print(f"Error: {e}")
            if "limit" in str(e):
                time.sleep(10)  # Adjust sleep if API limit is reached
            else:
                break  # Stop processing this instance on critical error
    return {"instance": instance, "all_responses": all_responses}


def main(args):
    summeval = json.load(open(args.summeval_fp))
    # summeval = summeval[:5]
    prompt = open(args.prompt_fp).read()

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        future_to_instance = {
            executor.submit(process_instance, inst, prompt, args.model, args.n): inst
            for inst in summeval
        }
        new_json = []
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_instance), total=len(summeval)
        ):
            result = future.result()
            if result and "all_responses" in result:
                new_json.append(result)
            else:
                print("Failed to process an instance.")

    with open(args.save_fp, "w") as f:
        json.dump(new_json, f, indent=4)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt_fp", type=str, required=True)
    argparser.add_argument("--save_fp", type=str, required=True)
    argparser.add_argument("--max_tokens", type=int, default=200)
    argparser.add_argument(
        "--summeval_fp", type=str, default="data/summeval copie.json"
    )
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--n", type=int)
    args = argparser.parse_args()

    main(args)
