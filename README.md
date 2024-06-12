### Data Expansion via Diffusion and LLMs for Effective Clothes-Changing Person Re-ID
### Dataset Link:
https://huggingface.co/datasets/ihaveamoose/DLCR/tree/main

The generated datasets maintain the same directory structure as the corresponding original datasets. For each original image, we created a .png file that includes all 10 variants of the original image. For example, given the original sample:

![image](https://github.com/CroitoruAlin/dlcr/assets/37226076/280bbd50-dfd4-4a39-8219-fede5edc34a1)

We store its 10 variants as a single image of the following form:

![image](https://github.com/CroitoruAlin/dlcr/assets/37226076/11f3bb5f-96ce-4366-82fc-aec476c16b7b)

Additionally, we store a .txt file that contains the descriptions of the clothing for each variant, with each description on a separate line.
```
Gray shirt| Blue jeans| Black shoes
Black top| Black pants| Black shoes
White top| Blue jeans| White shoes
blue shirt| black shorts| black shoes.
Shirt | Jeans | Shoes
Yellow t-shirt| Black pants| Black shoes
Gray top|Blue jeans|White shoes
Green top| Blue shorts| White shoes
Gray t-shirt| Black pants| Black shoes
Black shirt| Black pants| Black shoes
```
To read the generated images when you have acces to the name of the original image, you can use the following code:
```bash
generated_images = os.path.join(self.generated_data_location, img_name+".png")
generated_images = Image.open(generated_images)
generated_images = np.array(generated_images)
generated_images = rearrange(generated_images, 'h (b w) c->b h w c', b=10)
```
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
