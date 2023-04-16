"""GUI joint control test application."""

import argparse

import numpy as np
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient

from ..hand_body import HandBody
from ..hand_model_axislayer import HandModel22, HandModel45

parser = argparse.ArgumentParser("GUI debug tool")
parser.add_argument("--dofs", type=int, default=22, help="Number of degrees of freedom (22 or 45)")

parser.add_argument("--left-hand", dest="left_hand", action="store_true", help="Show left hand")
parser.add_argument("--right-hand", dest="left_hand", action="store_false", help="Show right hand")
parser.set_defaults(left_hand=False)

parser.add_argument("--visual-shapes", dest="visual", action="store_true", help="Show visual shapes")
parser.add_argument("--no-visual-shapes", dest="visual", action="store_false", help="Hide visual shapes")
parser.set_defaults(visual=True)

parser.add_argument("--self-collisions", dest="self_collisions", action="store_true", help="Enable self collisions")
parser.add_argument(
    "--no-self-collisions", dest="self_collisions", action="store_false", help="Disable self collisions"
)
parser.set_defaults(self_collisions=False)


def main(args):
    """Test GUI application."""
    client = BulletClient(pb.GUI)
    client.setGravity(0, 0, -10)

    client.resetDebugVisualizerCamera(
        cameraDistance=0.5, cameraYaw=-40.0, cameraPitch=-40.0, cameraTargetPosition=[0.0, 0.0, 0.0]
    )

    if args.dofs == 22:
        hand_model = HandModel22(left_hand=args.left_hand)
    elif args.dofs == 45:
        hand_model = HandModel45(left_hand=args.left_hand)
    else:
        raise ValueError("Only 22 and 45 DoF models are supported.")

    flags = sum(
        [
            HandBody.FLAG_ENABLE_COLLISION_SHAPES,
            HandBody.FLAG_ENABLE_VISUAL_SHAPES * args.visual,
            HandBody.FLAG_JOINT_LIMITS,
            HandBody.FLAG_DYNAMICS,
            HandBody.FLAG_USE_SELF_COLLISION * args.self_collisions,
        ]
    )

    client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
    hand = HandBody(client, hand_model, flags=flags)
    client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    slider_ids = []
    for coord in ("X", "Y", "Z"):
        uid = client.addUserDebugParameter(f"base_{coord}", -0.5, 0.5, 0)
        slider_ids.append(uid)
    for coord in ("R", "P", "Y"):
        uid = client.addUserDebugParameter(f"base_{coord}", -3.14, 3.14, 0)
        slider_ids.append(uid)
    for i, joint in enumerate(hand_model.joints):
        name = hand_model.link_names[i]
        for axis, (lower, upper) in zip(joint.axes, joint.limits):
            uid = client.addUserDebugParameter(f"{name}[{axis}]", lower, upper, 0)
            slider_ids.append(uid)

    client.setRealTimeSimulation(True)

    # * Dump
    # from pybullet_utils import urdfEditor

    # parser = urdfEditor.UrdfEditor()
    # parser.initializeFromBulletBody(hand.body_id, physicsClientId=client._client)
    # parser.saveUrdf("hand.urdf")
    # *

    try:
        while client.isConnected():
            values = [client.readUserDebugParameter(uid) for uid in slider_ids]

            position = values[0:3]
            rotation = client.getQuaternionFromEuler(values[3:6])
            angles = values[6:]

            # hand.set_target(position, rotation, angles)

            import pickle

            # pickle.dump(angles, open("angles.pkl", "wb"))
            # aa = pickle.load(open("~/Downloads/composed_aa.pkl", "rb"))
            # mano_pose = hand_model.angles_to_mano(np.array(angles))
            new_aa = hand_model.angles_to_mano(np.array(angles))
            new_angles, _ = hand_model.mano_to_angles(new_aa)
            assert np.allclose(new_angles, np.array(angles), atol=1e-6)

            hand.set_target(position, rotation, new_angles.tolist())

            # assert np.allclose(np.array(angles), new_angles, atol=1e-6)
    except pb.error as err:
        if str(err) not in ["Not connected to physics server.", "Failed to read parameter."]:
            raise


if __name__ == "__main__":
    main(parser.parse_args())
