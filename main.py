import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import VGG19_Weights
import time

from model import *

def preprocess(device, content_path, style_path, img_size=1024):
    content_image = Image.open(content_path).convert('RGB')
    style_image = Image.open(style_path).convert('RGB')

    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor()])

    content_image = transform(content_image).unsqueeze(0)
    style_image = transform(style_image).unsqueeze(0)

    content_image = content_image.to(device, dtype=torch.float)
    style_image = style_image.to(device, dtype=torch.float)

    assert content_image.size() == style_image.size(), "style and content image must be same size"
    return content_image, style_image

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=100000, content_weight=1):

    print("building style transfer model...")
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean,
                                                                    normalization_std, style_img, content_img,)

    input_img.requires_grad_(True)

    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.001)

    print("optimizing...")
    run = [0]
    while run[0] < num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0,1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 1 == 0:
                print('run {}'.format(run))
                print('style loss: {:.4f} content loss: {:.4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)
        scheduler.step()

    with torch.no_grad():
        input_img.clamp(0,1)

    return input_img

def main():
    device = torch.device("mps")
    torch.set_default_device(device)

    content_path = "./input/raccoon_foreground.jpg"
    style_path = "./input/blue.jpg"

    scales = [256, 512, 1024]
    output_img = None

    cnn = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    start_time = time.time()
    for img_size in scales:
        print(f"Processing scale: {img_size}")

        if output_img is None:
            content_img, style_img = preprocess(device, content_path, style_path, img_size)
            input_img = content_img.clone()
        else:
            output_cpu = output_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            output_image = (output_cpu * 255).clip(0, 255).astype("uint8")
            output_pil = Image.fromarray(output_image)

            content_img, style_img = preprocess(device, content_path, style_path, img_size)
            output_pil = transforms.Resize((img_size, img_size))(output_pil)

            input_img = transforms.ToTensor()(output_pil).unsqueeze(0).to(device)

        print(f"Input image size for the model: {input_img.shape}")

        style_weight = 10000
        num_steps = 400
        output_img = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                        content_img, style_img, input_img, style_weight=style_weight, num_steps=num_steps)
        
        output_cpu = output_img.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        output_image = (output_cpu * 255).clip(0, 255).astype("uint8")
        output_pil = Image.fromarray(output_image)

        output_pil.save(f"output/blue/output_{img_size}_{style_weight}_{num_steps}.png")

    total_time = time.time() - start_time
    print(total_time)
    
    # original_width, original_height = Image.open(content_path).convert('RGB').size
    # output_pil.resize((original_width, original_height), Image.LANCZOS)
    # output_pil.save("output/output_album2.png")

if __name__ == "__main__":
    main()
