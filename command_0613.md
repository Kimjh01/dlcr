[command 정리_0613]

cd LLaVA
python extract_clothes_descriptions.py -s "../../data/001_01" --model_path "./llava-v1.5-7b" --output_file ./prcc_clothes_descriptions3.jsonl


cd ../llama
summarize_clothes_descriptions.py 안에 parser 바꿔줘야함
python summarize_clothes_descriptions.py 


cd ../Self-Correction-Human-Parsing
python simple_extractor.py --dataset lip --model-restore ./exp-schp-201908261155-lip.pth --input-dir ../../data/001_01 --output-dir ./prcc_masks3


cd ../stable-diffusion
python generate_data.py --original_images_path ../../data/001_01 --clothes_description_path ../llama/clothes_descriptions3.json --masks_path ../Self-Correction-Human-Parsing/prcc_masks3/001_01/001_01 --output_directory_path ./generated_data3


cd ../DG
@click.option('--savedir', help='Save directory', metavar='PATH', type=str, required=True,
              default="/pretrained_models3/discriminator3")
@click.option('--gendir', help='Fake sample absolute directory', metavar='PATH', type=str, required=True,
              default="../stable-diffusion/generated_data3/prcc")
@click.option('--pretrained_classifier_ckpt', help='Path of ADM classifier', metavar='STR', type=str,
              default='/./pretrained_models3/32x32_classifier.pt')
이 부분 바꾸기

python train.py --datadir ../../data/001_01


cd ../stable-diffusion

parser.add_argument("--classifier_ckpt", required=False, type=str, default=None)
parser.add_argument("--discriminator_ckpt", required=False, type=str, default=None)
파서에 이거 추가

python generate_data.py --original_images_path ../../data/001_01 --masks_path ../Self-Correction-Human-Parsing/prcc_masks3/001_01/001_01 --output_directory_path ./generated_data_0613-2 --clothes_description_path ../llama/parsed_clothes3.json --use_discriminator True --classifier_ckpt .../DG/pretrained_models3/discriminator3/classifier_120.pt --discriminator_ckpt .../DG/pretrained_models3/discriminator3/discriminator_120.pt