import vgg_loss
import torch
import os

from torch.nn import functional as F
from PIL import Image


def apply_batch_mask(batch, instance_masks):
    max_masks = torch.max(
        instance_masks, axis=1
    ).values
    return batch * max_masks


def apply_mask(model_pred, target, latents, instance_masks):
    model_pred_prior, model_pred = torch.chunk(model_pred, 2, dim=0)
    target_prior, target = torch.chunk(target, 2, dim=0)
    latents_prior, latents = torch.chunk(latents, 2, dim=0)

    max_masks = torch.max(
        instance_masks, axis=1
    ).values
    downsampled_mask = F.interpolate(
        input=max_masks, size=(64, 64)
    )
    model_pred = model_pred * downsampled_mask
    model_pred_prior = model_pred_prior * downsampled_mask
    target = target * downsampled_mask
    target_prior = target_prior * downsampled_mask
    latents = latents * downsampled_mask
    latents_prior = latents_prior * downsampled_mask

    return (
        torch.cat([model_pred_prior, model_pred]),
        torch.cat([target_prior, target]),
        torch.cat([latents_prior, latents]),
    )
#------new--- for applying Mask to VGG results
# def VGG_Mask (imagesT, instance_masks):
#    Vgg_instance_Masks=instance_masks
#    max_masks = torch.max(
#         Vgg_instance_Masks, axis=1
#     ).values
#    downsampled_mask = F.interpolate(
#         input=max_masks, size=(imagesT.size())
#     )
#    Vgg_instance_Masks = Vgg_instance_Masks * downsampled_mask
#---------
    


def compute_vgg_loss(model_pred, latents, timesteps, target, instance_masks, batch, dreambooth):
    if dreambooth.args.with_vgg_loss_and_mask:
        # apply mask
        model_pred, target, latents = apply_mask(model_pred, target, latents, instance_masks)
        batch = apply_batch_mask(batch, instance_masks)

    imagesT = latent2img(model_pred, latents, timesteps, dreambooth)

    if dreambooth.args.with_image_target:
        batch = latent2img(target, latents, timesteps, dreambooth)

    crit_vgg = vgg_loss.VGGLoss().to(imagesT.device)

    return crit_vgg(imagesT.float(), batch.float(), target_is_features=False)


def debug_images(model_pred, latents, timesteps, target, instance_masks, batch, dreambooth):
    with torch.no_grad():
        imagesT = latent2img(model_pred, latents, timesteps, dreambooth)
        save_img(batch, timesteps, 'batch', dreambooth)
        save_img(imagesT, timesteps, 'imagesT', dreambooth)

        model_pred_masked, target_masked, latents_masked = apply_mask(model_pred, target, latents, instance_masks)

        imagesTarget = latent2img(target_masked, latents, timesteps, dreambooth)
        save_img(imagesTarget, timesteps, 'imagesTarget', dreambooth)

        imagesP = latent2img(model_pred_masked, latents, timesteps, dreambooth)
        save_img(imagesP, timesteps, 'imagesP', dreambooth)

        imagesLM = latent2img(model_pred_masked, latents_masked, timesteps, dreambooth)
        save_img(imagesLM, timesteps, 'imagesLM', dreambooth)

#-----new
        # imageVggMask =latent2img(model_pred_masked, latents_masked, timesteps, dreambooth)
        # save_img(imagesLM, timesteps, 'imagesLM', dreambooth)
#---------

def latent2img(model_pred, latents, timesteps, dreambooth, guidance_scale=7.5):
    # FIXME fix a bug on validation scheduler
    dreambooth.validation_scheduler.alphas_cumprod = dreambooth.validation_scheduler.alphas_cumprod.to(timesteps.device)
    # /FIXME

    latents = latents[-1][None]

    noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
    model_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
    )
    latentsT = dreambooth.validation_scheduler.step(model_pred, timesteps, latents).prev_sample
    latentsT = 1 / 0.18215 * latentsT

    #---- the latent passed to decoder ,,, 
    imagesT = dreambooth.vae.decode(latentsT.to(dreambooth.weight_dtype)).sample

    return imagesT


def save_img(imagesT, timesteps, img_name, dreambooth):
    imagesT_out = (imagesT / 2 + 0.5).clamp(0, 1)
    imagesT_out = imagesT_out.detach().cpu().permute(0, 2, 3, 1).numpy()
    imagesT_out = (imagesT_out * 255).round().astype("uint8")
    img_logs_path = os.path.join(dreambooth.args.output_dir, "img_timesteps")
    os.makedirs(img_logs_path, exist_ok=True)
    path = '{}/{}_{}.jpg'.format(img_logs_path, img_name, timesteps)
    Image.fromarray(imagesT_out[0]).save(path)