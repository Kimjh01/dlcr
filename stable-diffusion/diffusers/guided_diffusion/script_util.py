from .unet import EncoderUNetModel

NUM_CLASSES = 1000
def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    out_channels,
    in_channels = 4,
    condition=False,
):

    if 32 in image_size:
        channel_mult = (1, 2, 3, 4)
    else:
        channel_mult = (1,)
    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append((image_size[0] // int(res), image_size[1] // int(res)))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=classifier_width,
        out_channels=out_channels,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
        condition=condition,
    )

