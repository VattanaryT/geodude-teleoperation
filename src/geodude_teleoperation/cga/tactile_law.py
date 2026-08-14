"""Eq. 56 contact torque law (coverage citation).

τ = −J_{ee,G}^⊤ · (K_{L} W_L − D_V W_V + W_C + W_tan)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geodude_teleoperation.cga.line_tracker import (
    alignment_twist_se3,
    alignment_wrench_from_twist,
)
from geodude_teleoperation.cga.wrenches import (
    ContactWrenchPID,
    damping_wrench_from_qdot,
    desired_contact_wrench_ee,
    tangent_wrench_from_world_force,
    wrench_ee_to_base,
    wrench_to_ee_frame,
)
from geodude_teleoperation.teleop.tactile_impedance import joint_torques_from_wrench


@dataclass(frozen=True)
class Eq56Gains:
    """Diagonal stiffness / damping in wrench coordinates ``[f; m]``."""

    K_line: np.ndarray  # (6,)
    D_twist: np.ndarray  # (6,)


def default_eq56_gains() -> Eq56Gains:
    """Research defaults for stable kitchenware-style interaction."""
    return Eq56Gains(
        K_line=np.array([30.0, 30.0, 30.0, 750.0, 750.0, 300.0]),
        D_twist=np.array([10.0, 10.0, 10.0, 150.0, 150.0, 50.0]),
    )


def assemble_eq56_tau(
    jacobian_ee: np.ndarray,
    *,
    W_L: np.ndarray,
    W_V: np.ndarray,
    W_C: np.ndarray,
    W_tan: np.ndarray | None = None,
    gains: Eq56Gains | None = None,
) -> np.ndarray:
    """Assemble joint torques from Eq. 56 wrench sum (no gravity — DTC compensates).

    All wrenches must share the Jacobian frame (base/spatial at TCP).

    ``W_L`` / ``W_V`` are geometric error / damping (enter with paper signs).
    ``W_C`` / ``W_tan`` are contact / teleop wrenches the EE should **apply to the
    environment**, so they enter with opposite sign inside the sum:
    ``τ = −Jᵀ (K W_L − D W_V − W_C − W_tan) = −Jᵀ(K W_L − D W_V) + Jᵀ(W_C+W_tan)``.
    """
    g = gains or default_eq56_gains()
    K = np.asarray(g.K_line, dtype=float).reshape(6)
    D = np.asarray(g.D_twist, dtype=float).reshape(6)
    W_sum = (
        K * np.asarray(W_L, dtype=float).reshape(6)
        - D * np.asarray(W_V, dtype=float).reshape(6)
        - np.asarray(W_C, dtype=float).reshape(6)
    )
    if W_tan is not None:
        W_sum = W_sum - np.asarray(W_tan, dtype=float).reshape(6)
    return joint_torques_from_wrench(jacobian_ee, W_sum)


@dataclass
class Eq56Controller:
    """Stateful Eq. 56 tick: line align + normal-force PID + damping + tangent FF."""

    gains: Eq56Gains
    force_pid: ContactWrenchPID
    desired_force_n: float = 5.0
    force_sign: float = 1.0
    wrench_in_tool_frame: bool = True
    tangent_force_gain: float = 50.0
    max_tau: float = 20.0
    contact_threshold_n: float = 0.6
    approach_force_n: float = 8.0
    max_wrench_n: float = 12.0
    overshoot_margin_n: float = 2.0
    retract_force_n: float = 8.0

    def __init__(
        self,
        *,
        gains: Eq56Gains | None = None,
        desired_force_n: float = 5.0,
        force_sign: float = 1.0,
        wrench_in_tool_frame: bool = True,
        wrench_kp: float = 0.25,
        wrench_ki: float = 0.0,
        wrench_kd: float = 0.0,
        tangent_force_gain: float = 50.0,
        max_tau: float = 20.0,
        contact_threshold_n: float = 0.6,
        approach_force_n: float = 8.0,
        max_wrench_n: float = 12.0,
        overshoot_margin_n: float = 2.0,
        retract_force_n: float = 8.0,
    ):
        self.gains = gains or default_eq56_gains()
        self.force_pid = ContactWrenchPID(
            kp=float(wrench_kp), ki=float(wrench_ki), kd=float(wrench_kd)
        )
        self.desired_force_n = float(desired_force_n)
        self.force_sign = float(force_sign)
        self.wrench_in_tool_frame = bool(wrench_in_tool_frame)
        self.tangent_force_gain = float(tangent_force_gain)
        self.max_tau = float(max_tau)
        self.contact_threshold_n = float(contact_threshold_n)
        self.approach_force_n = float(approach_force_n)
        self.max_wrench_n = float(max_wrench_n)
        self.overshoot_margin_n = float(overshoot_margin_n)
        self.retract_force_n = float(retract_force_n)
        self.last_diag: dict = {}

    def reset(self) -> None:
        self.force_pid.reset()
        self.last_diag = {}

    def step(
        self,
        *,
        p_tcp: np.ndarray,
        R_tcp: np.ndarray,
        x_q: np.ndarray,
        R_field: np.ndarray,
        wrench_meas6: np.ndarray,
        jacobian_ee: np.ndarray,
        qdot: np.ndarray,
        v_tan_world: np.ndarray | None,
        dt: float,
        enable_force: bool = True,
    ) -> np.ndarray:
        V = alignment_twist_se3(p_tcp, R_tcp, x_q, R_field)
        W_L = alignment_wrench_from_twist(V)
        W_V = damping_wrench_from_qdot(jacobian_ee, qdot)

        W_m = wrench_to_ee_frame(
            wrench_meas6,
            R_tcp,
            wrench_in_tool_frame=self.wrench_in_tool_frame,
        )
        W_d = desired_contact_wrench_ee(
            desired_force_n=self.desired_force_n,
            force_sign=self.force_sign,
        )
        # Compression scalar: positive when pushing into the surface.
        fs_comp = float(self.force_sign) * float(W_m[2])
        fd = abs(float(self.desired_force_n))
        W_approach = np.zeros(6, dtype=float)
        overshoot = 0.0
        if enable_force:
            if fs_comp > fd + float(self.overshoot_margin_n):
                # Unload only — do not keep driving into an overshoot.
                overshoot = 1.0
                W_C = np.zeros(6, dtype=float)
                W_C[2] = -float(self.force_sign) * max(
                    float(self.retract_force_n), fs_comp - fd
                )
                self.force_pid.reset()
            else:
                W_C = self.force_pid.step(W_d, W_m, dt)
                # Soft approach only while still unloaded (signed compression).
                if fs_comp < float(self.contact_threshold_n):
                    f_app = max(
                        float(self.approach_force_n), abs(float(self.desired_force_n))
                    )
                    W_approach[2] = float(self.force_sign) * f_app
                    W_C = W_C + W_approach
            wn = float(np.linalg.norm(W_C[:3]))
            w_cap = max(float(self.max_wrench_n), fd)
            if wn > w_cap and wn > 1e-12:
                W_C[:3] = W_C[:3] * (w_cap / wn)
        else:
            W_C = np.zeros(6, dtype=float)

        W_tan_ee_or_base = np.zeros(6, dtype=float)
        if v_tan_world is not None and overshoot < 0.5:
            f_world = float(self.tangent_force_gain) * np.asarray(
                v_tan_world, dtype=float
            ).reshape(3)
            W_tan_ee_or_base = tangent_wrench_from_world_force(f_world, R_tcp)

        W_C_base = wrench_ee_to_base(W_C, R_tcp)
        W_tan_base = W_tan_ee_or_base
        W_approach_base = wrench_ee_to_base(W_approach, R_tcp)

        tau = assemble_eq56_tau(
            jacobian_ee,
            W_L=W_L,
            W_V=W_V,
            W_C=W_C_base,
            W_tan=W_tan_base,
            gains=self.gains,
        )
        tn = float(np.linalg.norm(tau))
        if tn > self.max_tau and tn > 1e-12:
            tau = tau * (self.max_tau / tn)

        self.last_diag = {
            "W_L_norm": float(np.linalg.norm(W_L)),
            "W_V_norm": float(np.linalg.norm(W_V)),
            "W_C_norm": float(np.linalg.norm(W_C_base)),
            "W_approach_norm": float(np.linalg.norm(W_approach_base)),
            "W_tan_norm": float(np.linalg.norm(W_tan_base)),
            "tau_norm": float(np.linalg.norm(tau)),
            "V_align_norm": float(np.linalg.norm(V)),
            "Fs_ee_z": float(W_m[2]),
            "Fs_comp": float(fs_comp),
            "Fd_N": float(self.desired_force_n),
            "force_enable": 1.0 if enable_force else 0.0,
            "force_overshoot": float(overshoot),
            "f_cmd_base_x": float(W_C_base[0]),
            "f_cmd_base_y": float(W_C_base[1]),
            "f_cmd_base_z": float(W_C_base[2]),
        }
        return tau
