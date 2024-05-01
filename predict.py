# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)

from ppgan.utils.options import parse_args
from ppgan.utils.config import get_config
from ppgan.utils.setup import setup
from ppgan.engine.trainer import Trainer
from ppgan.datasets.preprocess.io import LoadImageFromFile
from PIL import Image
import paddle
from ops import split_image, concat_image
from ppgan.models.generators.builder import build_generator

def preprocess(path):

    load = LoadImageFromFile()


def main(args, cfg):
    setup(args, cfg)
    trainer = Trainer(cfg)
    img = Image.open("images/3.jpeg")
    ims, (nh, nw, h, w) = split_image(img, size=(32, 32))

    weight = paddle.load("output_dir/esrgan_psnr_x4_div2k-2024-05-01-16-32/iter_110000_weight.pdparams")
    trainer.model.nets['generator'].set_state_dict(weight['generator'])

    trainer.model.nets['generator'].eval()
    print(trainer.model.nets['generator'].training)

    with paddle.no_grad():
        outs = trainer.model.nets['generator'](ims)

    print(outs.shape, outs.min(), outs.max())
    img = concat_image(outs, nh, nw, h*4, w*4)
    img.save("./output/1.png")
    print("保存成功")


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.config_file, args.opt)


    main(args, cfg)
