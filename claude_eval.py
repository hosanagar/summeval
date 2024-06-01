from anthropic import Anthropic
import json
import argparse
import tqdm
import time

# Include system prompt (!)

client = Anthropic(api_key="")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt_fp", type=str, required=True)
    argparser.add_argument("--save_fp", type=str, required=True)
    argparser.add_argument("--summeval_fp", type=str, default="data/summeval.json")
    argparser.add_argument("--model", type=str, default="claude-3-haiku-20240307")
    args = argparser.parse_args()

    print(args.summeval_fp)
    summeval = json.load(open(args.summeval_fp))
    prompt = open(args.prompt_fp).read()

    ct, ignore = 0, 0

    new_json = []
    n = 20 if args.model == "claude-3-haiku-20240307" else 5
    for instance in tqdm.tqdm(summeval):
        source = instance["source"]
        system_output = instance["system_output"]
        cur_prompt = prompt.replace("{{Document}}", source).replace(
            "{{Summary}}", system_output
        )
        instance["prompt"] = cur_prompt
        while True:
            try:
                all_responses = []
                for i in range(n):
                    _response = client.messages.create(
                        model=args.model,
                        messages=[{"role": "user", "content": cur_prompt}],
                        temperature=1,
                        max_tokens=200,
                    )
                    time.sleep(0.5)
                    response = _response.content[0].text
                    all_responses.append(response)
                instance["all_responses"] = all_responses
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(e)
                if "limit" in str(e):
                    time.sleep(2)
                else:
                    ignore += 1
                    print("ignored", ignore)
                    if ignore > 5:
                        break
        if ignore > 4 or ct > 4:
            break

    print("ignored total", ignore)
    with open(args.save_fp, "w") as f:
        json.dump(new_json, f, indent=4)
