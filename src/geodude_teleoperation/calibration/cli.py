"""CLI for UR force/torque sensor zeroing (TCP/payload: use teach pendant)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import yaml

from geodude_teleoperation.calibration import ft_sensor
from geodude_teleoperation.calibration.io import (
    VALID_ARMS,
    arm_yaml_path,
    default_calib_dir,
    load_arm_calib,
    merge_section,
    parse_float_list,
    save_arm_calib,
)
from geodude_teleoperation.calibration.rtde_session import RtdeSession


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ConnectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geodude-calibrate",
        description=(
            "Sample / zero the UR FT sensor and record results in YAML. "
            "TCP and payload are set on the teach pendant, not by this tool."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, *, need_apply_flags: bool = False) -> None:
        p.add_argument("--arm", required=True, choices=VALID_ARMS)
        p.add_argument(
            "--calib-dir",
            type=Path,
            default=None,
            help="Directory for <arm>.yaml (default: repo calibration/)",
        )
        p.add_argument("--ip", default=None, help="Robot IP (else YAML robot_ip)")
        if need_apply_flags:
            p.add_argument(
                "--apply",
                action="store_true",
                help="Call zeroFtSensor() on the controller",
            )
            p.add_argument(
                "--yes",
                action="store_true",
                help="Skip zero confirmation prompt",
            )

    p_ft = sub.add_parser("ft", help="Sample FT bias and optionally zero sensor")
    add_common(p_ft, need_apply_flags=True)
    p_ft.add_argument("--samples", type=int, default=100)
    p_ft.add_argument("--note", default="")
    p_ft.add_argument(
        "--bias",
        default=None,
        help="Optional bias wrench Fx,Fy,Fz,Tx,Ty,Tz (skip live sampling)",
    )
    p_ft.set_defaults(func=_cmd_ft)

    p_show = sub.add_parser("show", help="Print saved calibration YAML")
    add_common(p_show)
    p_show.set_defaults(func=_cmd_show)

    return parser


def _calib_dir(args: argparse.Namespace) -> Path:
    return Path(args.calib_dir) if args.calib_dir is not None else default_calib_dir()


def _path_for(args: argparse.Namespace) -> Path:
    return arm_yaml_path(_calib_dir(args), args.arm)


def _confirm(message: str, yes: bool) -> bool:
    if yes:
        return True
    reply = input(f"{message} [y/N]: ").strip().lower()
    return reply in ("y", "yes")


def _resolve_ip(args: argparse.Namespace, data: dict) -> str:
    if args.ip:
        return str(args.ip)
    if data.get("robot_ip"):
        return str(data["robot_ip"])
    raise ValueError("robot IP required: pass --ip or set robot_ip in the YAML file")


def _with_session(ip: str, fn: Callable[[RtdeSession], None]) -> None:
    try:
        with RtdeSession(ip) as session:
            fn(session)
    except ImportError:
        raise
    except Exception as exc:
        raise RuntimeError(f"RTDE operation failed for {ip}: {exc}") from exc


def _cmd_ft(args: argparse.Namespace) -> int:
    path = _path_for(args)
    existing = load_arm_calib(path)
    bias: list[float]
    zeroed = False

    if args.bias is not None:
        bias = parse_float_list(args.bias, 6)
        if args.apply:
            ip = _resolve_ip(args, existing)
            if _confirm(f"apply zeroFtSensor() to {ip}?", args.yes):
                _with_session(ip, ft_sensor.apply_ft_zero)
                zeroed = True
            else:
                print("zero skipped")
    else:
        ip = _resolve_ip(args, existing)
        holder: dict = {}
        try:
            with RtdeSession(ip) as session:
                holder["bias"] = ft_sensor.sample_ft_bias(session, samples=args.samples)
                if args.apply:
                    if _confirm(f"apply zeroFtSensor() to {ip}?", args.yes):
                        ft_sensor.apply_ft_zero(session)
                        zeroed = True
                    else:
                        print("zero skipped")
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(f"FT sampling failed for {ip}: {exc}") from exc
        bias = holder["bias"]

    section = ft_sensor.build_ft_section(
        bias, sample_count=args.samples, note=args.note
    )
    data = merge_section(existing, args.arm, "ft_sensor", section, robot_ip=args.ip)
    save_arm_calib(path, data)
    print(f"saved ft_sensor -> {path}")
    if zeroed:
        print(f"zeroed FT sensor on {_resolve_ip(args, data)}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    path = _path_for(args)
    data = load_arm_calib(path)
    if not data:
        print(f"(empty) {path}")
        return 0
    print(yaml.safe_dump(data, default_flow_style=False, sort_keys=False).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
