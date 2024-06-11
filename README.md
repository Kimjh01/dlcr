### Data Expansion via Diffusion and LLMs for Effective Clothes-Changing Person Re-ID
### Dataset Link:
https://ucf-my.sharepoint.com/:f:/r/personal/ny525072_ucf_edu/Documents/DLCR?csf=1&web=1&e=Feeo2l
#### Instructions
1. Run LLaVA

```bash
cd LLaVA
```
Follow the setup instructions described in the README.md file.
Run the clothes descriptions with the following command
```bash
python extract_clothes_descriptions.py -s <path_training_data> --model_path <path_checkpoint> --output_file ./prcc_clothes_descriptions.jsonl
```
Check also the README file for downloading the model weights.

2. Create the summaries using llama:

```bash
cd ../llama
```
Follow the setup instructions described in the README.md file.
Extract the summaries:
```bash
torchrun --nproc_per_node 1 summarize_clothes_descriptions.py     --ckpt_dir Llama-2-7b-chat/     --tokenizer_path ./Llama-2-7b-chat/tokenizer.model     --max_seq_len 512 --max_batch_size 6
```
3. Run mask extraction
```bash
cd ../Self-Correction-Human-Parsing

```
Follow the setup instructions described in the README.md file
```bash
python simple_extractor.py --dataset lip --model-restore exp-schp-201908261155-lip.pth --input-dir <path_training_data> --output-dir prcc_masks
```
4. Generate data for discriminator training:
```bash
cd ../stable-diffusion
```
Follow the setup instructions from the README.md file
```bash
python generate_data.py --original_images_path <path_training_data> --masks_path ../Self-Correction-Human-Parsing/prcc_masks --output_directory_path . --clothes_description_path ../llama/parsed_clothes.json

```
5. Train the discriminator
```bash
cd ../DG
```
Follow the setup instructions from the README.md file
```bash
python train.py --datadir <path_training_data>
```
6. Rerun the generation, but this time with the discriminator on

```bash
cd ../stable-diffusion
python generate_data.py --original_images_path <path_training_data> --masks_path ../Self-Correction-Human-Parsing/prcc_masks --output_directory_path . --clothes_description_path ../llama/parsed_clothes.json --use_discriminator True

```
7. Train the CC-Reid model

```
cd ../Simple-CCReID
```
Follow the setup instructions from the README.md file
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0 --root <dir_containing_data> --gen_path <path_generated_data>
```
After training is done, the prediction refinement method can be run:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main_evaluation.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0 --root <dir_containing_data> --gen_path <path_test_generated_data>
```
The prediction refinement requires new variants for the queries. Thus, before running the last command, it is needed to generate these variants.
You can do this by rerunning steps 3 and 4, but this time the input data should be the queries.

Test link
