"""Hand models."""

import collections

import numpy as np
import os
from .mano_model import ManoModel
from .math_utils import joint2mat, mat2joint, mat2rvec, rvec2mat
import torch
from manotorch.axislayer import AxisAdaptiveLayer, AxisLayerFK
from manotorch.manolayer import ManoLayer
from manotorch.utils.geometry import rotation_to_axis_angle, axis_angle_to_matrix

__all__ = ("HandModel", "HandModel22", "HandModel45")

Joint = collections.namedtuple("Joint", ["origin", "basis", "axes", "limits"])


class HandModel(ManoModel):
    """Base rigid hand model.

    The model provides rigid hand kinematics description and math_utils
    from the joint space to the MANO model pose.
    """

    def __init__(self, left_hand=False, models_dir=None):
        """Initialize a HandModel.

        Keyword Arguments:
            left_hand {bool} -- create a left hand model (default: {False})
            models_dir {str} -- path to the pickled model files (default: {None})
        """
        super().__init__(left_hand=left_hand, models_dir=models_dir)

        if models_dir is None:
            models_dir = os.path.expandvars("$MANO_MODELS_DIR")

        self.manolayer = ManoLayer(
            side="left" if left_hand else "right",
            mano_assets_root=models_dir.replace("/models", ""),
        )
        self.axislayer = AxisLayerFK(
            side="left" if left_hand else "right",
            mano_assets_root=models_dir.replace("/models", ""),
        )

        self._joints = self._make_joints()
        self._basis = [joint.basis for joint in self._joints]
        self._axes = [joint.axes for joint in self._joints]
        self._dofs = [
            (u - len(self._axes[i]), u) for i, u in enumerate(np.cumsum([len(joint.axes) for joint in self._joints]))
        ]

        assert len(self._joints) == len(self.origins()), "Wrong joints number"
        assert all([len(j.axes) == len(j.limits) for j in self._joints]), "Wrong limits number"
        assert not self._joints[0].axes, "Palm joint is not fixed"

    @property
    def joints(self):
        """Joint descriptions.

        Returns:
            list -- list of Joint structures
        """
        return self._joints

    @property
    def dofs_number(self):
        """Number of degrees of freedom.

        Returns:
            int -- sum of degrees of freedom of all joints
        """
        return sum([len(joint.axes) for joint in self._joints[1:]])

    @property
    def dofs_limits(self):
        """Limits corresponding to degrees of freedom.

        Returns:
            tuple -- lower limits list, upper limits list
        """
        return list(zip(*[limits for joint in self._joints[1:] for limits in joint.limits]))

    def angles_to_mano(self, angles, palm_basis=None):
        """Convert joint angles to a MANO pose.

        Arguments:
            angles {array} -- rigid model's dofs angles

        Keyword Arguments:
            palm_basis {mat3} -- palm basis (default: {None})

        Returns:
            array -- MANO pose, array of size N*3 where N - number of links
        """
        if len(angles) != self.dofs_number:
            raise ValueError(f"Expected {self.dofs_number} angles (got {len(angles)}).")

        euler_angles = np.zeros((16, 3), dtype=np.float32)
        euler_angles[1][1] = angles[0]
        euler_angles[1][2] = angles[1]
        euler_angles[2][2] = angles[2]
        euler_angles[3][2] = angles[3]
        euler_angles[4][1] = angles[4]
        euler_angles[4][2] = angles[5]
        euler_angles[5][2] = angles[6]
        euler_angles[6][2] = angles[7]
        euler_angles[7][1] = angles[8]
        euler_angles[7][2] = angles[9]
        euler_angles[8][2] = angles[10]
        euler_angles[9][2] = angles[11]
        euler_angles[10][1] = angles[12]
        euler_angles[10][2] = angles[13]
        euler_angles[11][2] = angles[14]
        euler_angles[12][2] = angles[15]
        euler_angles[13][0] = angles[16]
        euler_angles[13][1] = angles[17]
        euler_angles[13][2] = angles[18]
        euler_angles[14][1] = angles[19]
        euler_angles[14][2] = angles[20]
        euler_angles[15][2] = angles[21]

        mano_pose = self.axislayer.compose(torch.tensor(euler_angles)[None])

        return mano_pose.numpy()[0]

    def mano_to_angles(self, hand_pose):
        """Convert a mano pose to joint angles of the rigid model.

        It is not guaranteed that the rigid model can ideally
        recover a mano pose.

        Arguments:
            mano_pose {array} -- MANO pose, array of size N*3 where N - number of links

        Returns:
            tuple -- dofs angles, palm_basis
        """

        mano_pose = hand_pose.reshape((-1, 3)).copy()
        global_rot = hand_pose[0].copy()
        mano_pose[0] = np.zeros(3)

        mano_out = self.manolayer(torch.tensor(mano_pose).reshape(1, 16 * 3), torch.zeros(1, 10))
        transf = mano_out.transforms_abs  # [1, 16, 4, 4]
        _, _, ee_a_tmplchd_chd = self.axislayer(transf)
        ee_a_tmplchd_chd = ee_a_tmplchd_chd[0]
        angles = np.zeros(22)
        angles[0] = ee_a_tmplchd_chd[1][1]
        angles[1] = ee_a_tmplchd_chd[1][2]
        angles[2] = ee_a_tmplchd_chd[2][2]
        angles[3] = ee_a_tmplchd_chd[3][2]
        angles[4] = ee_a_tmplchd_chd[4][1]
        angles[5] = ee_a_tmplchd_chd[4][2]
        angles[6] = ee_a_tmplchd_chd[5][2]
        angles[7] = ee_a_tmplchd_chd[6][2]
        angles[8] = ee_a_tmplchd_chd[7][1]
        angles[9] = ee_a_tmplchd_chd[7][2]
        angles[10] = ee_a_tmplchd_chd[8][2]
        angles[11] = ee_a_tmplchd_chd[9][2]
        angles[12] = ee_a_tmplchd_chd[10][1]
        angles[13] = ee_a_tmplchd_chd[10][2]
        angles[14] = ee_a_tmplchd_chd[11][2]
        angles[15] = ee_a_tmplchd_chd[12][2]
        angles[16] = ee_a_tmplchd_chd[13][0]
        angles[17] = ee_a_tmplchd_chd[13][1]
        angles[18] = ee_a_tmplchd_chd[13][2]
        angles[19] = ee_a_tmplchd_chd[14][1]
        angles[20] = ee_a_tmplchd_chd[14][2]
        angles[21] = ee_a_tmplchd_chd[15][2]

        return angles, axis_angle_to_matrix(torch.tensor(global_rot)).numpy()

    def _make_joints(self):
        """Compute joints parameters.

        Returns:
            list -- list of joints parameters
        """
        raise NotImplementedError


class HandModel22(HandModel):
    """Heuristic rigid model with 22 degrees of freedom."""

    def _make_joints(self):
        """Compute joints parameters.

        Returns:
            list -- list of joints parameters
        """
        # origin = dict(zip(self.link_names, self.origins()))

        # from termcolor import cprint

        # cprint(origin, "cyan")
        # basis = {"palm": np.eye(3)}

        # def make_basis(yvec, zvec):
        #     mat = np.vstack([np.cross(yvec, zvec), yvec, zvec])
        #     return mat.T / np.linalg.norm(mat.T, axis=0)

        # zvec = origin["index2"] - origin["index3"]
        # yvec = np.cross(zvec, [0.0, 0.0, 1.0])
        # basis["index"] = make_basis(yvec, zvec)

        # zvec = origin["middle2"] - origin["middle3"]
        # yvec = np.cross(zvec, origin["index1"] - origin["ring1"])
        # basis["middle"] = make_basis(yvec, zvec)

        # zvec = origin["ring2"] - origin["ring3"]
        # yvec = np.cross(zvec, origin["middle1"] - origin["ring1"])
        # basis["ring"] = make_basis(yvec, zvec)

        # zvec = origin["pinky2"] - origin["pinky3"]
        # yvec = np.cross(zvec, origin["ring1"] - origin["pinky1"])
        # basis["pinky"] = make_basis(yvec, zvec)

        # yvec = origin["thumb1"] - origin["index1"]
        # zvec = np.cross(yvec, origin["thumb1"] - origin["thumb2"])
        # basis["thumb0"] = make_basis(yvec, zvec)

        # zvec = origin["thumb2"] - origin["thumb3"]
        # yvec = np.cross(zvec, [0, -np.sin(0.96), np.cos(0.96)])
        # basis["thumb"] = make_basis(yvec, zvec)

        if self.is_left_hand:
            # rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            # basis = {key: mat @ rot for key, mat in basis.items()}
            raise NotImplementedError()

        hand_part = [
            "palm",
            "index1",
            "index2",
            "index3",
            "middle1",
            "middle2",
            "middle3",
            "pinky1",
            "pinky2",
            "pinky3",
            "ring1",
            "ring2",
            "ring3",
            "thumb1",
            "thumb2",
            "thumb3",
        ]

        origin = {}
        basis = {}

        for k in range(16):
            origin[hand_part[k]] = self.axislayer.TMPL_T_g_a[0][k][:3, 3].numpy()
            basis[hand_part[k]] = self.axislayer.TMPL_T_g_a[0][k][:3, :3].numpy()  # [[2, 1, 0]]

        # * y: up axis
        # * z: left axis

        return [
            Joint(origin["palm"], basis["palm"], "", []),
            Joint(
                origin["index1"],
                basis["index1"],
                "yz",
                np.deg2rad([(-10, 10), (0, 90)]),
            ),
            Joint(origin["index2"], basis["index2"], "z", np.deg2rad([(0, 100)])),
            Joint(origin["index3"], basis["index3"], "z", np.deg2rad([(0, 80)])),
            Joint(
                origin["middle1"],
                basis["middle1"],
                "yz",
                np.deg2rad([(-10, 10), (0, 90)]),
            ),
            Joint(origin["middle2"], basis["middle2"], "z", np.deg2rad([(0, 100)])),
            Joint(origin["middle3"], basis["middle3"], "z", np.deg2rad([(0, 80)])),
            Joint(
                origin["pinky1"],
                basis["pinky1"],
                "yz",
                np.deg2rad([(-10, 10), (0, 90)]),
            ),
            Joint(origin["pinky2"], basis["pinky2"], "z", np.deg2rad([(0, 100)])),
            Joint(origin["pinky3"], basis["pinky3"], "z", np.deg2rad([(0, 80)])),
            Joint(origin["ring1"], basis["ring1"], "yz", np.deg2rad([(-10, 10), (0, 90)])),
            Joint(origin["ring2"], basis["ring2"], "z", np.deg2rad([(0, 100)])),
            Joint(origin["ring3"], basis["ring3"], "z", np.deg2rad([(0, 80)])),
            Joint(
                origin["thumb1"],
                basis["thumb1"],
                "xyz",
                np.deg2rad([(0, 45), (-15, 45), (-45, 45)]),
            ),
            Joint(
                origin["thumb2"],
                basis["thumb2"],
                "yz",
                np.deg2rad([(-10, 10), (0, 90)]),
            ),
            Joint(origin["thumb3"], basis["thumb3"], "z", np.deg2rad([(0, 80)])),
        ]


class HandModel45(HandModel):
    """Rigid model with 45 degrees of freedom."""

    def _make_joints(self):
        """Compute joints parameters.

        Returns:
            list -- list of joints parameters
        """
        origin = dict(zip(self.link_names, self.origins()))
        limits = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi, np.pi)]

        return [
            Joint(origin["palm"], np.eye(3), "", []),
            Joint(origin["index1"], np.eye(3), "xyz", limits),
            Joint(origin["index2"], np.eye(3), "xyz", limits),
            Joint(origin["index3"], np.eye(3), "xyz", limits),
            Joint(origin["middle1"], np.eye(3), "xyz", limits),
            Joint(origin["middle2"], np.eye(3), "xyz", limits),
            Joint(origin["middle3"], np.eye(3), "xyz", limits),
            Joint(origin["pinky1"], np.eye(3), "xyz", limits),
            Joint(origin["pinky2"], np.eye(3), "xyz", limits),
            Joint(origin["pinky3"], np.eye(3), "xyz", limits),
            Joint(origin["ring1"], np.eye(3), "xyz", limits),
            Joint(origin["ring2"], np.eye(3), "xyz", limits),
            Joint(origin["ring3"], np.eye(3), "xyz", limits),
            Joint(origin["thumb1"], np.eye(3), "xyz", limits),
            Joint(origin["thumb2"], np.eye(3), "xyz", limits),
            Joint(origin["thumb3"], np.eye(3), "xyz", limits),
        ]
