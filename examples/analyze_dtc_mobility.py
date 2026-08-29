"""Why did the arm move *that* way? -- model-free and model-based, from a dtc log.

Gain tuning answers "is it stable"; it cannot answer "why does wrist3 take the
motion instead of the shoulder". That question is decided by mobility -- how
much motion one N.m buys at each joint -- and by whether the achieved EE twist
points where the commanded wrench pointed. Both are measurable from a log that
already exists, with no hardware and no experiment.

Section 1 needs numpy only. Sections 2-4 need gafropy (robot machine).

    python3 examples/analyze_dtc_mobility.py [log.csv] [--task-space right_ee]
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newest_log():
    hits = sorted(glob.glob(os.path.join(ROOT, "speedj_logs", "dtc_*.csv")),
                  key=os.path.getmtime)
    if not hits:
        sys.exit("no speedj_logs/dtc_*.csv found")
    return hits[-1]


def load(path):
    rows = list(csv.DictReader(open(path)))
    n = sum(1 for k in rows[0] if k.startswith("tau"))
    g = lambda p: np.array([[float(r[f"{p}{i}"]) for i in range(n)] for r in rows])
    t = np.array([float(r["t"]) for r in rows])
    dt = np.array([float(r["dt_ms"]) / 1e3 if r["dt_ms"] else np.nan for r in rows])
    w = np.array([[float(r[f"w{i}"]) for i in range(6)] for r in rows])
    return t, dt, w, g("tau"), g("q"), g("qd"), n


def section1_mobility(t, dt, tau, q, qd, n):
    """Model-free: rad of travel bought per N.m.s of applied torque, per joint.

    This is the number that decides which joint absorbs a task-space wrench, and
    it is a physical property of the arm at this pose and payload -- stiction,
    gravity load and gearing all fold into it. No gain changes it. A joint two
    decades more mobile than another will take essentially all of the motion
    whenever the jacobian gives it any share of the torque at all.
    """
    print("\n1. EMPIRICAL MOBILITY  (model-free)")
    dq = np.abs(np.diff(q, axis=0))
    tt = np.abs(tau[1:])
    step = np.where(np.isfinite(dt[1:]), dt[1:], np.nanmedian(dt))
    impulse = tt * step[:, None]
    mob, tot_dq, tot_imp = [], [], []
    for j in range(n):
        m = tt[:, j] > 0.25          # ignore ticks where the joint is barely driven
        d, i = dq[m, j].sum(), impulse[m, j].sum()
        tot_dq.append(d)
        tot_imp.append(i)
        mob.append(d / i if i > 1e-9 else np.nan)
    mob = np.array(mob)
    ref = np.nanmin(mob[mob > 0]) if np.any(mob > 0) else np.nan
    print("   joint   travel(rad)  impulse(N.m.s)   mobility(rad/N.m.s)   vs stiffest")
    for j in range(n):
        rel = mob[j] / ref if np.isfinite(mob[j]) and np.isfinite(ref) and ref > 0 else np.nan
        print(f"    j{j}      {tot_dq[j]:8.3f}     {tot_imp[j]:9.2f}        "
              f"{mob[j]:10.4f}          {rel:7.1f}x")
    if np.isfinite(ref):
        print(f"\n   spread = {np.nanmax(mob)/ref:.0f}x. Above ~10x, a task-space wrench with "
              f"no inertia\n   weighting (tau = J.T @ W) is decided by mobility, not by the "
              f"wrench direction.")


def section2_alignment(w, tau, q, qd, n, task_space_name, model):
    """Model-based: did the arm move where the wrench pointed?"""
    try:
        import gafropy
        from geodude_teleoperation.torque_backend import (
            resolve_control_group_slots, splice_ur)
        from geodude_teleoperation.vention import vention_dof_indices
    except Exception as exc:
        print(f"\n2-4. skipped (needs gafropy on the robot machine): {exc}")
        return
    system = gafropy.SystemSerialization.load(model)
    ts = system.get_task_space(task_space_name)
    ji = list(ts.get_joint_indices())
    # .values(): vention_dof_indices is name -> index, and resolve_ wants the
    # indices. Same call the backend makes (torque_backend.py:444).
    ur_slots, _ = resolve_control_group_slots(
        ji, vention_dof_indices(system).values(),
        ts.get_controlled_joint_indices())
    base = np.zeros(len(ji))
    ur_global = np.asarray(ji, dtype=int)[np.asarray(ur_slots, dtype=int)]

    cos_lin, ang_frac_cmd, ang_frac_got, tau_lin, tau_ang = [], [], [], [], []
    for k in range(0, len(q), max(1, len(q) // 4000)):
        if np.linalg.norm(w[k, 3:]) < 1.0:
            continue
        qc = splice_ur(base, q[k], ur_slots)
        qdc = splice_ur(np.zeros(base.size), qd[k], ur_slots)
        J = np.asarray(ts.compute_state(qc, qdc).get_jacobian(), dtype=float)
        xd = J @ qd[k]
        nl, na = np.linalg.norm(xd[3:]), np.linalg.norm(xd[:3])
        if nl > 1e-6:
            c = float(w[k, 3:] @ xd[3:] / (np.linalg.norm(w[k, 3:]) * nl))
            cos_lin.append(c)
        if nl + na > 1e-6:
            ang_frac_got.append(na / (nl + na))
        wl, wa = np.linalg.norm(w[k, 3:]), np.linalg.norm(w[k, :3])
        if wl + wa > 1e-9:
            ang_frac_cmd.append(wa / (wl + wa))
        tau_lin.append(np.abs(J.T[:, 3:] @ w[k, 3:]))
        tau_ang.append(np.abs(J.T[:, :3] @ w[k, :3]))

    if not cos_lin:
        print("\n2-4. no ticks with a real linear command; nothing to compare")
        return
    print("\n2. DID IT GO WHERE IT WAS PUSHED?")
    c = np.array(cos_lin)
    print(f"   cos(W_lin, achieved EE linear velocity): median {np.median(c):+.3f}  "
          f"frac>0.7: {(c > 0.7).mean():.2f}  frac<0: {(c < 0).mean():.2f}")
    print("   ~+1 = the TCP goes where you push (mapping is right).")
    print("   ~0  = the motion is orthogonal to the command -- kinematics, not gains.")
    print("   <0  = a frame or sign is inverted.")

    print("\n3. TRANSLATION vs ROTATION")
    print(f"   commanded angular share: {np.median(ang_frac_cmd):.2f}")
    print(f"   achieved  angular share: {np.median(ang_frac_got):.2f}")
    print("   achieved >> commanded = the error is escaping as rotation.")

    print("\n4. WHICH JOINT IS ASKED TO DO THE TRANSLATING  (mean |tau| from each blade)")
    tl, ta = np.mean(tau_lin, axis=0), np.mean(tau_ang, axis=0)
    print("   joint   from W_lin   from W_ang")
    for j in range(n):
        print(f"    j{j}      {tl[j]:8.2f}     {ta[j]:8.2f}")
    print("   Cross this with section 1: a joint with a large share AND low "
          "mobility\n   is where your force is disappearing.")
    section5_lambda(system, ts, q, qd, ur_slots, ur_global, base)


def section5_lambda(system, ts, q, qd, ur_slots, ur_global, base):
    """Preflight: is Lambda safe to command torque with, at EVERY pose in this log?

    Sampling a dozen ticks is not a safety check. Lambda = (J M^-1 J.T)^-1 is the
    factor the commanded force is multiplied by, so a single near-singular pose
    is a single large uncommanded push -- and that is what a sparse sample
    misses. This sweeps the whole log and reports how often Lambda leaves the
    band the arm physically has, which is the number that decides whether
    --dtc-lambda may be used at all.

    Hand-rolled from System.compute_mass_matrix; the built-in
    compute_task_space_inertia_matrix was measured at ~1e14 on this model.
    """
    from geodude_teleoperation.torque_backend import (
        splice_ur, LAMBDA_MAX_LIN_KG, LAMBDA_MAX_ANG_KGM2,
        LAMBDA_SIGMA_FULL, LAMBDA_SIGMA_NONE)
    print("\n5. LAMBDA PREFLIGHT  (every tick)")
    try:
        n_full = int(np.asarray(system.compute_mass_matrix(
            np.zeros(len(system.get_joints())))).shape[0])
    except Exception as exc:
        print(f"   cannot size the model: {exc}")
        return
    lin_hi, ang_hi, sig, n_bad, n_fail = [], [], [], 0, 0
    for k in range(len(q)):
        try:
            qc = splice_ur(base, q[k], ur_slots)
            qdc = splice_ur(np.zeros(base.size), qd[k], ur_slots)
            J = np.asarray(ts.compute_state(qc, qdc).get_jacobian(), dtype=float)
            s_min = float(np.sqrt(max(0.0, float(
                np.linalg.eigvalsh(J @ J.T)[0]))))
            q_full = np.zeros(n_full)
            q_full[ur_global] = q[k]
            M = np.asarray(system.compute_mass_matrix(q_full), dtype=float)
            Mi = np.linalg.inv(M[np.ix_(ur_global, ur_global)])
            Lam = np.linalg.inv(J @ Mi @ J.T)
            l = np.linalg.eigvalsh(0.5 * (Lam[3:, 3:] + Lam[3:, 3:].T)).max()
            a = np.linalg.eigvalsh(0.5 * (Lam[:3, :3] + Lam[:3, :3].T)).max()
        except Exception:
            n_fail += 1
            continue
        lin_hi.append(l); ang_hi.append(a); sig.append(s_min)
        if l > LAMBDA_MAX_LIN_KG or a > LAMBDA_MAX_ANG_KGM2 or not np.isfinite(l + a):
            n_bad += 1
    if not lin_hi:
        print("   Lambda could not be built on any tick")
        return
    lin_hi, ang_hi, sig = map(np.array, (lin_hi, ang_hi, sig))
    n = len(lin_hi)
    print(f"   linear  max eigenvalue (kg):      median {np.median(lin_hi):8.2f}"
          f"  p99 {np.percentile(lin_hi, 99):10.2f}  max {lin_hi.max():12.2f}")
    print(f"   angular max eigenvalue (kg.m2):   median {np.median(ang_hi):8.4f}"
          f"  p99 {np.percentile(ang_hi, 99):10.4f}  max {ang_hi.max():12.4f}")
    print(f"   sigma_min of J:                   median {np.median(sig):8.4f}"
          f"  p01 {np.percentile(sig, 1):10.4f}  min {sig.min():12.4f}")
    print(f"   ticks past the clamp ({LAMBDA_MAX_LIN_KG:g} kg / "
          f"{LAMBDA_MAX_ANG_KGM2:g} kg.m2): {n_bad}/{n} = {100.0*n_bad/n:.1f}%")
    print(f"   ticks inside the singularity fade (<{LAMBDA_SIGMA_FULL:g}): "
          f"{100.0*(sig < LAMBDA_SIGMA_FULL).mean():.1f}%   "
          f"fully faded out (<{LAMBDA_SIGMA_NONE:g}): "
          f"{100.0*(sig < LAMBDA_SIGMA_NONE).mean():.1f}%")
    if n_fail:
        print(f"   {n_fail} ticks could not build Lambda at all")
    frac = n_bad / n
    print()
    if frac > 0.02:
        print(f"   NO-GO: Lambda leaves its physical band on {100*frac:.1f}% of poses in")
        print("   this log. The clamp will be active there, so the shaped force is")
        print("   bounded but no longer the true inertia -- and the poses doing it are")
        print("   near-singular. Fix the workspace or the reach before commanding torque.")
    else:
        print(f"   GO: Lambda stays inside its band on {100*(1-frac):.1f}% of poses.")
    print("\n   NOTE: M(q) comes from geodude.yaml and does NOT include a payload")
    print("   calibrated on the teach pendant. If a custom tool is fitted, Lambda is")
    print("   wrong by that tool's mass and inertia even where it is well conditioned.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--task-space", default="right_ee")
    ap.add_argument("--model", default=os.path.join(
        ROOT, "assets", "robots", "geodude", "geodude.yaml"))
    a = ap.parse_args()
    path = a.log or newest_log()
    t, dt, w, tau, q, qd, n = load(path)
    print(f"{os.path.relpath(path, ROOT)}: {len(t)} ticks, {n} joints, {t[-1]:.1f} s")
    section1_mobility(t, dt, tau, q, qd, n)
    section2_alignment(w, tau, q, qd, n, a.task_space, a.model)


if __name__ == "__main__":
    main()
