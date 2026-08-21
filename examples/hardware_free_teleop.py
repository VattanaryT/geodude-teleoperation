import argparse
import pathlib
import socket
import time

import gafropy as ga
import numpy as np
import yaml
from gafropy import robot_interface as ri
from gafropy import viz as gafropy_viz

from geodude_teleoperation import (
    AdmittanceTeleoperationController,
    KeyboardTeleop,
    SpaceMouseTeleop,
    TeleoperationController,
    DEFAULT_VENTION_HEIGHT,
    apply_vention_heights,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = str(ROOT / "assets" / "robots" / "geodude" / "geodude.yaml")
DEFAULT_HARDWARE_CONFIG = str(ROOT / "assets" / "robots" / "geodude" / "geodude_hardware.yaml")


def _dashboard_bring_up(ip: str) -> None:
    sock = socket.create_connection((ip, 29999), timeout=5)
    sock.recv(4096)

    def command(text: str) -> str:
        sock.sendall((text + "\n").encode())
        time.sleep(0.3)
        return sock.recv(4096).decode(errors="replace")

    command("power on")
    time.sleep(2)
    command("brake release")
    for _ in range(30):
        mode = command("robotmode")
        if "RUNNING" in mode:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"{ip}: robot never reached RUNNING mode")
    sock.close()
    print(f"{ip}: RUNNING")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task-space", choices=["left_ee", "right_ee"], default="left_ee")
    parser.add_argument(
        "--device", choices=["keyboard", "spacemouse"], default="spacemouse")
    parser.add_argument(
        "--controller", choices=["velocity", "admittance"], default="admittance")
    parser.add_argument(
        "--hardware-config", default=DEFAULT_HARDWARE_CONFIG,
        help="YAML file mapping control groups to driver configs (Configuration.hardware_config)")
    parser.add_argument(
        "--no-bring-up", dest="bring_up", action="store_false",
        help="skip the dashboard power-on/brake-release sequence (already-running real hardware)")
    parser.add_argument("--no-viz", action="store_true",
                        help="disable the viser visualization")
    parser.add_argument(
        "--vention-height", type=float, default=DEFAULT_VENTION_HEIGHT,
        help="physical vention rail height in meters for both arms (hardware does "
             "not sense these joints; used for visualization FK)")
    parser.add_argument("--left-vention-height", type=float, default=None)
    parser.add_argument("--right-vention-height", type=float, default=None)
    args = parser.parse_args()

    if args.bring_up:
        with open(args.hardware_config) as f:
            hardware_config = yaml.safe_load(f)
        for control_group, entry in hardware_config.items():
            _dashboard_bring_up(entry["robot_ip"])

    config = ri.Configuration(MODEL, "hardware", args.hardware_config)
    interface = ri.RobotInterface(config)
    interface.initialize()

    left_vention = args.left_vention_height if args.left_vention_height is not None else args.vention_height
    right_vention = args.right_vention_height if args.right_vention_height is not None else args.vention_height
    print(f"hardware does not report vention rails; drawing them at "
          f"left={left_vention:.3f}m right={right_vention:.3f}m")

    def q_with_vention():
        return apply_vention_heights(
            interface.get_system(), interface.get_state().get_position(),
            left_vention, right_vention)

    task_space = interface.get_task_space(args.task_space)
    if task_space is None:
        raise RuntimeError(f"task space '{args.task_space}' not found in {MODEL}")

    joint_idx = np.array(task_space.get_joint_indices())
    live_position = interface.get_state().get_position()[joint_idx]
    home_ee_motor = task_space.compute_ee_motor(live_position)

    controller = (
        TeleoperationController(damping=0.05) if args.controller == "velocity"
        else AdmittanceTeleoperationController()
    )
    controller.set_task_space(task_space)
    controller.set_interface(interface)
    controller.set_objective(
        ga.control.SingleArmMotorControlObjective(home_ee_motor))

    interface.control(controller, blocking=False)

    def on_twist(twist):
        controller.update_command(twist)

    def on_button(index):
        print("no button")

    device = KeyboardTeleop(
        step=0.0025) if args.device == "keyboard" else SpaceMouseTeleop()
    device.start(on_twist, on_button)

    try:
        if args.no_viz:
            print(f"Teleoperating '{args.task_space}' on real geodude (headless, "
                  f"free space, {args.controller} controller). Ctrl+C to quit.")
            while True:
                time.sleep(0.1)
        else:
            viz = gafropy_viz.Visualizer()

            robot = viz.add_robot(interface.get_system(),
                                  name="geodude", joint_sliders=False)
            robot.update(q_with_vention())

            eeviz = viz.add_motor(home_ee_motor)

            def update():
                q = q_with_vention()
                robot.update(q)
                eeviz.update(task_space.compute_ee_motor(q[joint_idx]))

            viz.add_ticker(lambda: update())

            viz.show(message=f"Teleoperating '{args.task_space}' on real geodude "
                     f"(free space, {args.controller} controller). "
                     f"viser running at http://localhost:{viz.port} -- ctrl-c to quit.")
    except KeyboardInterrupt:
        pass
    finally:
        device.stop()
        interface.shutdown()


if __name__ == "__main__":
    main()
