from math import pi

import numpy as np
import torch
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
from urdfpy import URDF

from mano_pybullet.hand_model_axislayer import HandModel22

mano_layer = ManoLayer(
    rot_mode="axisang", center_idx=9, mano_assets_root="assets/mano_v1_2", use_pca=False, flat_hand_mean=True
)

axisFK = AxisLayerFK(mano_assets_root="assets/mano_v1_2")
composed_ee = torch.zeros((1, 16, 3))

#         15-14-13-\
#                   \
#   3-- 2 -- 1 ----- 0
#   6 -- 5 -- 4 ----/
#   12 - 11 - 10 --/
#    9-- 8 -- 7 --/

composed_ee[:, 1:4] = torch.tensor([[0, 0, pi / 3]] * 3).unsqueeze(0)
composed_ee[:, 13:16] = torch.tensor([[0, 0, pi / 4]] * 3).unsqueeze(0)

composed_aa = axisFK.compose(composed_ee).clone()
composed_aa = composed_aa.reshape(1, -1)
zero_shape = torch.zeros((1, 10))

mano_output: MANOOutput = mano_layer(composed_aa, zero_shape)

urdf_hand = HandModel22(models_dir="assets/mano_v1_2/models")

# ! NOTE: You have to handle the global rotation manually

# * from mano parameters to urdf
urdf, global_rot = urdf_hand.mano_to_angles(composed_aa.reshape(16, 3).numpy())
# * from urdf to mano parameters
mano_aa = urdf_hand.angles_to_mano(urdf)
# * should be the same
assert np.allclose(mano_aa, composed_aa.reshape(16, 3).numpy(), atol=1e-7)

robot = URDF.load("urdf/mano.urdf")
key_list = [
    "j_index1y",
    "j_index1z",
    "j_index2",
    "j_index3",
    "j_middle1y",
    "j_middle1z",
    "j_middle2",
    "j_middle3",
    "j_pinky1y",
    "j_pinky1z",
    "j_pinky2",
    "j_pinky3",
    "j_ring1y",
    "j_ring1z",
    "j_ring2",
    "j_ring3",
    "j_thumb1x",
    "j_thumb1y",
    "j_thumb1z",
    "j_thumb2y",
    "j_thumb2z",
    "j_thumb3",
]

robot.animate(
    cfg_trajectory={k: [0, v] for k, v in zip(key_list, urdf)},
)
