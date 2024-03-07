import argparse
import os
import json

from llava.eval.model_vqa import eval_model


def create_questions(args):
    # prcc
    q_id = 1
    root_dir = args.source_dir
    if os.path.exists("questions.jsonl"):
        os.remove("questions.jsonl")
    for dir in os.listdir(root_dir):
        if 'json' in dir:
            continue
        for image in os.listdir(os.path.join(root_dir, dir)):
            entry = {"question_id": q_id, "image": image, "text": "What clothes is the person wearing?",
                     "category": "conv", 'dir': dir}
            q_id += 1
            str_entry = json.dumps(entry)
            with open('questions.jsonl', 'a') as the_file:
                the_file.write(f'{str_entry}\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-s", "--source_dir", required=True, type=str,
                    help="Input data path")
    ap.add_argument("-mp", '--model_path', required=True, type=str,
                    help="Model path")
    ap.add_argument("-o", "--output_file", required=True, type=str,
                    help="Path to output file. If this exists, it will be overwritten and if it doesn't it will be created")

    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--conv-mode", type=str, default="llava_v1")
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--chunk-idx", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--num_beams", type=int, default=1)
    args = ap.parse_args()
    create_questions(args)
    # create args for vqa
    args.question_file = "questions.jsonl"
    args.image_folder = args.source_dir
    args.answers_file = args.output_file
    eval_model(args)



