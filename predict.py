import os
import argparse
import torch
import json
import numpy
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models import CompletionNetwork
from PIL import Image
from utils import (
    process_image,
    process_image_test,
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
)


parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default="results/single_char_rotate_result/model_cn_step4500")
parser.add_argument('--config',type=str,default="results/single_char_rotate_result/config.json")
parser.add_argument('--input_dir',type=str,default="test_single/distort_input/")
parser.add_argument('--output_dir',type=str,default="test_single/distort_output/")
parser.add_argument('--img_size', type=int, default=256)


def main(args):



    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))


    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    image_list = os.listdir(args.input_dir)
    for image_name in image_list:
        img = Image.open(args.input_dir + image_name)
        img = img.resize((args.img_size, args.img_size), Image.ANTIALIAS)
        x = transforms.ToTensor()(img)
        x = torch.unsqueeze(x, dim=0)

        # inpaint
        with torch.no_grad():
            input = process_image_test(x)
            output = model(input)
            output = torch.cat((x, input, output), dim=-1)
            save_image(output, args.output_dir + image_name, nrow=3)
        print('distort_output img was saved as %s.' % args.output_dir + image_name)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
