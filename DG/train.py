import click
import os

from diffusers import StableDiffusionInpaintPipeline

import classifier_lib
import torch
import numpy as np
import dnnlib
from tqdm import tqdm
from guided_diffusion.image_datasets import load_data_latent
import random


@click.command()
@click.option('--savedir', help='Save directory', metavar='PATH', type=str, required=True,
              default="/pretrained_models/discriminator")
@click.option('--gendir', help='Fake sample absolute directory', metavar='PATH', type=str, required=True,
              default="../stable-diffusion/prcc")
@click.option('--datadir', help='Real sample absolute directory', metavar='PATH', type=str, required=True,
              default="real_latents_pad")
@click.option('--pretrained_classifier_ckpt', help='Path of ADM classifier', metavar='STR', type=str,
              default='/./pretrained_models/32x32_classifier.pt')
@click.option('--batch_size', help='Num samples', metavar='INT', type=click.IntRange(min=1), default=16)
@click.option('--epoch', help='Num samples', metavar='INT', type=click.IntRange(min=1), default=120)
@click.option('--lr', help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=3e-4)
@click.option('--device', help='Device', metavar='STR', type=str, default='cuda:0')
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    savedir = os.getcwd() + opts.savedir
    os.makedirs(savedir, exist_ok=True)

    ## Prepare fake/real data
    gen_train_loader = load_data_latent(
        data_dir=opts.gendir,
        batch_size=int(opts.batch_size / 2),
        image_size=(256, 768),
        class_cond=False,
        random_crop=False,
        random_flip=False,
    )
    real_train_loader = load_data_latent(
        data_dir=opts.datadir,
        batch_size=int(opts.batch_size / 2),
        image_size=(256, 768),
        class_cond=False,
        random_crop=False,
        random_flip=False,
    )

    ## Extractor & Disciminator
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        revision='fp16',
        device_map='auto',
        safety_checker=None,
        low_cpu_mem_usage=True
    )
    vae = pipe.vae
    pipe = None
    scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    # pretrained_classifier_ckpt = opts.pretrained_classifier_ckpt
    classifier = classifier_lib.load_classifier(None, [256 // scale_factor, 768 // scale_factor], opts.device, eval=False)
    discriminator = classifier_lib.load_discriminator(None, opts.device, False, eval=False)

    ## Prepare training
    vpsde = classifier_lib.vpsde()
    classifier_params = list(classifier.parameters())
    classifier_params.extend(discriminator.parameters())
    optimizer = torch.optim.Adam(classifier_params, lr=opts.lr, weight_decay=1e-7)
    loss = torch.nn.BCELoss()

    iterator = iter(gen_train_loader)
    ## Training
    for i in range(opts.epoch):
        outs = []
        cors = []
        num_data = 0
        for j, data in enumerate(tqdm(real_train_loader)):
            optimizer.zero_grad()
            real_inputs = data
            real_inputs = real_inputs.to(opts.device)
            with torch.no_grad():
                real_inputs = vae.encode(real_inputs.half()).latent_dist.sample().float()
            real_labels = torch.ones(real_inputs.shape[0]).to(opts.device)

            ## Real data perturbation
            real_t, _ = vpsde.get_diffusion_time(real_inputs.shape[0], opts.device, 1e-5, importance_sampling=True)
            mean, std = vpsde.marginal_prob(real_t)
            z = torch.randn_like(real_inputs)
            perturbed_real_inputs = mean[:, None, None, None] * real_inputs + std[:, None, None, None] * z

            ## Fake data
            try:
                fake_inputs = next(iterator)
            except:
                iterator = iter(gen_train_loader)
                fake_inputs = next(iterator)
            fake_inputs = fake_inputs.to(opts.device)
            fake_labels = torch.zeros(fake_inputs.shape[0]).to(opts.device)
            with torch.no_grad():
                fake_inputs = vae.encode(fake_inputs.half()).latent_dist.sample().float()
            ## Fake data perturbation
            fake_t, _ = vpsde.get_diffusion_time(fake_inputs.shape[0], opts.device, 1e-5, importance_sampling=True)
            mean, std = vpsde.marginal_prob(fake_t)
            z = torch.randn_like(fake_inputs)
            perturbed_fake_inputs = mean[:, None, None, None] * fake_inputs + std[:, None, None, None] * z

            ## Combine data
            inputs = torch.cat([real_inputs, fake_inputs])
            perturbed_inputs = torch.cat([perturbed_real_inputs, perturbed_fake_inputs])
            labels = torch.cat([real_labels, fake_labels])
            t = torch.cat([real_t, fake_t])
            c = list(range(inputs.shape[0]))
            random.shuffle(c)
            inputs, perturbed_inputs, labels, condition, t = inputs[c], perturbed_inputs[c], labels[c], None, t[
                c]

            ## Forward

            pretrained_feature = classifier(perturbed_inputs, timesteps=t, feature=True)

            label_prediction = discriminator(pretrained_feature, t, condition=condition, sigmoid=True).view(-1)
            label_prediction = torch.clip(label_prediction, min=1e-5, max=1 - 1e-5)
            # print(labels[:10], label_prediction[:10])

            ## Backward
            out = loss(label_prediction, labels)
            out.backward()
            optimizer.step()

            # label_prediction[label_prediction >= .5] = 1
            # label_prediction[label_prediction < .5] = 0

            # print(label_prediction[:20])
            accuracy = ((label_prediction > .5) == labels).float().mean()

            ## Report
            cor = ((label_prediction > 0.5).float() == labels).float().mean()
            outs.append(out.item())
            cors.append(cor.item())
            num_data += inputs.shape[0]
            if j == 25:
                print(f"{i}-th epoch BCE loss: {np.mean(outs)}, correction rate: {np.mean(cors)}, acc: {accuracy}")

            ## Save
            torch.save(discriminator.state_dict(), savedir + f"/discriminator_{i + 1}.pt")
            torch.save(classifier.state_dict(), savedir + f"/classifier_{i + 1}.pt")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------