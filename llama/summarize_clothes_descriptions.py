# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse
import json
import re
from typing import Optional

# import fire
from tqdm import tqdm
from llama import Llama
import time


def parse_answers(args):
    with open("clothes_descriptions.json") as file:
        clothes = json.load(file)

    new_clothes = {}
    for cloth_id in clothes:
        cloth_description = clothes[cloth_id]

        content = re.split("\\n[0-9]*.", cloth_description)[1:]

        cloth_items = content
        new_clothes[cloth_id] = cloth_items

    new_clothes = json.dumps(new_clothes)

    with open("parsed_clothes.json", "w") as file:
        file.write(new_clothes)


def group_clothes_descriptions(args):
    questions_file = args.llava_input_path
    answer_file = args.llava_output_path
    output_file = "grouped_clothes.json"
    with open(questions_file) as q_file:
        file_contents = q_file.read()

    questions = [json.loads(jline) for jline in file_contents.splitlines()]

    with open(answer_file) as a_file:
        file_contents = a_file.read()

    answers = [json.loads(jline) for jline in file_contents.splitlines()]

    dict_q = {}
    dict_a = {}
    for question in questions:
        dict_q[question['question_id']] = question
    for answer in answers:
        dict_a[answer['question_id']] = answer

    dict_clothes = {}
    for q_id in dict_a:
        id_person = dict_q[q_id]['dir']
        cloth_count = dict_q[q_id]['image'].split("_")[0]
        if cloth_count == 'A' or cloth_count == 'B':
            cloth_count = 'AB'
        id_cloth = f"{cloth_count}_{id_person}"
        if id_cloth not in dict_clothes:
            dict_clothes[id_cloth] = [dict_a[q_id]['text']]
        else:
            dict_clothes[id_cloth].append(dict_a[q_id]['text'])

    json_clothes = json.dumps(dict_clothes)
    with open(output_file, 'w') as the_file:
        the_file.write(json_clothes)


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
        max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    with open("grouped_clothes.json", "r") as f:
        content = f.read()
    json_file = json.loads(content)
    results = []
    dialogs = []
    for cloth_id in tqdm(json_file):
        text = ' '.join(json_file[cloth_id]).split(' ')[:200]
        dialog = [

            {"role": "user",
             "content": f"Extract the clothing items from the following text: {' '.join(text)}",
             "cloth_id": cloth_id}
        ]
        # print(dialog[0]['content'])
        dialogs.append(dialog)
        start_time = time.time()
        results.append(generator.chat_completion(
            [dialog],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ))
        end_time = time.time()
        print(f"Time taken:{end_time - start_time}")
    summaries = {}
    for dialog, result in zip(dialogs, results):
        print(result)
        summaries[dialog[0]['cloth_id']] = result[0]['generation']['content']
    clothes_summary_train = json.dumps(summaries)
    with open(args.output_file, 'w') as f:
        f.write(clothes_summary_train)


if __name__ == "__main__":
    # fire.Fire(main)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--max_batch_size", type=int)
    parser.add_argument("--output_file", default="clothes_descriptions.json", type=str)
    parser.add_argument("--llava_input_path", default="../LLaVA/questions.jsonl", type=str)
    parser.add_argument("--llava_output_path", default="../LLaVA/prcc_clothes_descriptions.jsonl", type=str)

    args = parser.parse_args()
    group_clothes_descriptions(args)
    main(ckpt_dir=args.ckpt_dir,
         tokenizer_path=args.tokenizer_path,
         max_seq_len=args.max_seq_len,
         max_batch_size=args.max_batch_size)
    parse_answers(args)
