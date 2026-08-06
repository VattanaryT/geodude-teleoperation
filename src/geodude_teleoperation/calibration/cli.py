"""CLI for UR TCP / payload / FT calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import yaml

from geodude_teleoperation.calibration import ft_sensor, payload, tcp
from geodude_teleoperation.calibration.io import (
    VALID_ARMS,
    arm_yaml_path,
    default_calib_dir,
    load_arm_calib,
    merge_section,
    parse_float_list,
    save_arm_calib,
    validate_payload,
    validate_tcp_pose,
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
        description="Calibrate UR TCP pose, payload, and FT sensor; save YAML and optionally apply via RTDE.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, *, need_apply_flags: bool = True) -> None:
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
                help="Push values to the robot via RTDE after saving",
            )
            p.add_argument(
                "--yes",
                action="store_true",
                help="Skip apply confirmation prompt",
            )

    p_tcp = sub.add_parser("tcp", help="Save TCP pose (manual entry)")
    add_common(p_tcp)
    p_tcp.add_argument(
        "--pose",
        default=None,
        help="x,y,z,rx,ry,rz (m / axis-angle). Prompted if omitted.",
    )
    p_tcp.set_defaults(func=_cmd_tcp)

    p_payload = sub.add_parser("payload", help="Save payload mass and CoG")
    add_common(p_payload)
    p_payload.add_argument("--mass", type=float, default=None, help="Payload mass in kg")
    p_payload.add_argument(
        "--cog",
        default=None,
        help="CoG x,y,z in meters relative to tool flange",
    )
    p_payload.set_defaults(func=_cmd_payload)

    p_ft = sub.add_parser("ft", help="Sample FT bias and optionally zero sensor")
    add_common(p_ft)
    p_ft.add_argument("--samples", type=int, default=100)
    p_ft.add_argument("--note", default="")
    p_ft.add_argument(
        "--bias",
        default=None,
        help="Optional bias wrench Fx,Fy,Fz,Tx,Ty,Tz (skip live sampling)",
    )
    p_ft.set_defaults(func=_cmd_ft)

    p_apply = sub.add_parser("apply", help="Apply saved sections to the robot")
    add_common(p_apply, need_apply_flags=False)
    p_apply.add_argument(
        "--what",
        default="tcp,payload,ft",
        help="Comma-separated sections: tcp,payload,ft",
    )
    p_apply.add_argument("--yes", action="store_true")
    p_apply.set_defaults(func=_cmd_apply)

    p_show = sub.add_parser("show", help="Print saved calibration YAML")
    add_common(p_show, need_apply_flags=False)
    p_show.set_defaults(func=_cmd_show)

    return parser


def _calib_dir(args: argparse.Namespace) -> Path:
    return Path(args.calib_dir) if args.calib_dir is not None else default_calib_dir()


def _path_for(args: argparse.Namespace) -> Path:
    return arm_yaml_path(_calib_dir(args), args.arm)


def _prompt(label: str) -> str:
    return input(f"{label}: ").strip()


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


def _cmd_tcp(args: argparse.Namespace) -> int:
    pose_text = args.pose if args.pose is not None else _prompt("TCP pose x,y,z,rx,ry,rz")
    pose = validate_tcp_pose(parse_float_list(pose_text, 6))
    section = tcp.build_tcp_section(pose)
    path = _path_for(args)
    data = merge_section(load_arm_calib(path), args.arm, "tcp", section, robot_ip=args.ip)
    save_arm_calib(path, data)
    print(f"saved tcp -> {path}")
    if args.apply:
        ip = _resolve_ip(args, data)
        if not _confirm(f"apply setTcp({pose}) to {ip}?", args.yes):
            print("apply skipped")
            return 0
        _with_session(ip, lambda s: tcp.apply_tcp(s, pose))
        print(f"applied tcp to {ip}")
    return 0


def _cmd_payload(args: argparse.Namespace) -> int:
    mass = args.mass
    cog_text = args.cog
    if mass is None:
        mass = float(_prompt("payload mass_kg"))
    if cog_text is None:
        cog_text = _prompt("payload cog x,y,z")
    mass, cog = validate_payload(mass, parse_float_list(cog_text, 3))
    section = payload.build_payload_section(mass, cog)
    path = _path_for(args)
    data = merge_section(load_arm_calib(path), args.arm, "payload", section, robot_ip=args.ip)
    save_arm_calib(path, data)
    print(f"saved payload -> {path}")
    if args.apply:
        ip = _resolve_ip(args, data)
        if not _confirm(f"apply setPayload({mass}, {cog}) to {ip}?", args.yes):
            print("apply skipped")
            return 0
        _with_session(ip, lambda s: payload.apply_payload(s, mass, cog))
        print(f"applied payload to {ip}")
    return 0


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
            # Do not update ft_sensor YAML on sample failure.
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


def _cmd_apply(args: argparse.Namespace) -> int:
    path = _path_for(args)
    data = load_arm_calib(path)
    if not data:
        raise ValueError(f"no calibration file at {path}")
    what = [w.strip() for w in str(args.what).split(",") if w.strip()]
    if not what:
        raise ValueError("--what must list at least one of tcp,payload,ft")
    section_map = {"tcp": "tcp", "payload": "payload", "ft": "ft_sensor"}
    missing = []
    for key in what:
        if key not in section_map:
            raise ValueError(f"unknown section {key!r}; expected tcp, payload, or ft")
        if section_map[key] not in data:
            missing.append(key)
    if missing:
        present = [k for k, sk in section_map.items() if sk in data]
        raise ValueError(
            f"missing section(s) {missing}; present: {present or 'none'}"
        )
    ip = _resolve_ip(args, data)
    summary = []
    if "tcp" in what:
        summary.append(f"tcp={data['tcp']['pose']}")
    if "payload" in what:
        summary.append(f"payload={data['payload']['mass_kg']},{data['payload']['cog']}")
    if "ft" in what:
        summary.append("ft=zeroFtSensor()")
    if not _confirm(f"apply to {ip}: {'; '.join(summary)}?", args.yes):
        print("apply skipped")
        return 0

    def _do(session: RtdeSession) -> None:
        if "tcp" in what:
            tcp.apply_tcp(session, data["tcp"]["pose"])
        if "payload" in what:
            payload.apply_payload(
                session, data["payload"]["mass_kg"], data["payload"]["cog"]
            )
        if "ft" in what:
            ft_sensor.apply_ft_zero(session)

    _with_session(ip, _do)
    print(f"applied {','.join(what)} to {ip}")
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
