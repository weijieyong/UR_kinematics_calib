### Proposed Project Structure

```
ur5-ik-project/                # Git root
│
├─ pyproject.toml             # managed by uv, declares deps (numpy, scipy,...)
├─ README.md                  # quick-start usage & badges
├─ CHANGELOG.md               # follow keep-a-changelog
│
├─ configs/                   # calibration config files extracted from UR robot
│   ├─ calibration.conf
│   └─ urcontrol.conf.UR5
│
├─ src/                       # import root → `python -m ur5_kinematics …`
│   ├─ dry_parse.py           # verbatim copy of existing file
│   └─ ur5_kinematics/        # new, fully-typed package
│       ├─ __init__.py        # re-exports FK/IK helpers
│       ├─ parameters.py      # thin loader that wraps dry_parse for calibrated values
│       ├─ fk.py              # optional convenience wrapper calling `dry_parse.fk_to_flange`
│       ├─ ik.py              # your new solver lives here
│       └─ util.py            # shared math, clipping, tolerance helpers
│
├─ scripts/                   # CLI entry-points (installed with `-m pip install -e .`)
│   ├─ fk_demo.py             # example: python scripts/fk_demo.py --q 0,-90,…
│   └─ ik_demo.py             # example: python scripts/ik_demo.py --pose 400,-200,…
│
├─ tests/                     # pytest suite (`uv venv pytest -q`)
│   ├─ test_fk_against_ur.py
│   ├─ test_ik_sanity.py
│   └─ fixtures/              # helper json/npz targets captured from the real robot
│
└─ docs/
    ├─ architecture.md        # high-level design
    ├─ ik_algorithm.md        # expanded maths & derivations
    └─ api_reference.md
```

*Why this layout?*

*   **`src/` layout** keeps importable code off the root `PYTHONPATH`, preventing shadowing during tests.
*   `dry_parse.py` is kept intact; new modules *import* it instead of duplicating its logic.
*   `parameters.py` centralises all *calibrated* quantities so every layer (FK, IK, tests) uses exactly the same numbers.
*   Scripts are thin wrappers that remain runnable after `pip install -e .`, while notebooks (if you need them) stay outside the package to avoid pulling heavy deps into the runtime image.

---

## Implementation Guide – Adding the IK Solver

Below is the recommended chain-of-thought. Follow the steps in order; each one builds on the previous.

> **Notation**
> *`θᵢ`* – joint *i* (robot convention: base→wrist)
> *`aᵢ, dᵢ, αᵢ`* – *calibrated* DH parameters (already corrected by `delta_*`).
> *`Tᵢⱼ`* – homogeneous transform from frame *i* to *j*.
> `T_base_tcp^target` – pose you want the tool flange to reach (4 × 4 matrix).

---

### 1 Bootstrap the new package

1.  `uv venv && uv pip install -e .` to enter a dev venv with an editable install.
2.  Inside `src/ur5_kinematics/__init__.py`, re-export:

    ```python
    from .parameters import get_calibrated_params, tcp_transform
    from .ik import solve_ik, IKResult
    ```
3.  Add *pytest*, *mypy*, *ruff* and *pre-commit* hooks to `pyproject.toml` to enforce style & typing.

---

### 2 Loader layer (`parameters.py`)

```python
"""Unified access to calibrated UR5 parameters."""
from pathlib import Path
import numpy as np
from . import dry_parse  # relative import; dry_parse sits in the same src tree

_project_root = Path(__file__).resolve().parents[2]  # ur5-ik-project/
_cfg_dir = _project_root / "configs"

def get_calibrated_params():
    c_path = _cfg_dir / "calibration.conf"
    u_path = _cfg_dir / "urcontrol.conf.UR5"
    dt, da, dd, dalpha = dry_parse.load_calibration(c_path)
    (a0, d0, alpha0, q_home, j_dirs), tcp_conf = dry_parse.load_urcontrol_config(u_path)
    a_eff, d_eff, alpha_eff = a0 + da, d0 + dd, alpha0 + dalpha
    T_fl_tcp = dry_parse.tcp_transform(tcp_conf)
    return a_eff, d_eff, alpha_eff, dt, q_home, j_dirs, T_fl_tcp
```

*This wrapper hides file paths & ensures every consumer uses identical numbers.*

---

### 3 Define IK outputs (`ik.py`)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class IKResult:
    q_rad: np.ndarray          # shape (6,)
    within_joint_limits: bool  # simple check; limits table lives in `util.py`
    pose_error_mm: float       # FK(θ) vs target TCP
```

Having a *structured* result makes later optimisation or logging trivial.

---

### 4 Choose the algorithm

UR5 has a **spherical wrist** (axes 4–6 intersect), so an *analytic closed-form* IK is both feasible and more performant than iterative methods:

1.  **Wrist-centre decoupling** – remove the TCP offset to find the centre of joint 5 (*p\_wc*).
2.  **First three joints (`θ₁-θ₃`)** – treat the arm as a planar 3-DoF mechanism; solve with geometry or law-of-cosines.
3.  **Last three joints (`θ₄-θ₆`)** – derive from the orientation of the spherical wrist (`R₃₆`).
4.  **Enumerate branches** – UR5 has 2×2×2 = 8 analytic solutions (shoulder left/right × elbow up/down × wrist flip/non-flip).
5.  **Apply calibration deltas & sign conventions** – *critical!* `dry_parse` already adds `delta_theta`; we must match its definition of `dh_thetas`:

    ```
    dh_theta_i = θ_i + delta_theta_i      (all values in radians)
    ```
6.  **Optionally refine** each candidate with one Newton–Raphson step if sub-0.01 mm accuracy is required.

---

### 5 Implement `solve_ik` (`ik.py`)

```python
def solve_ik(T_base_tcp_target: np.ndarray,
             return_first_valid: bool = False,
             tol_mm: float = 0.5,
             tol_rad: float = 1e-3) -> list[IKResult]:

    a, d, alpha, dt, _, joint_dirs, T_fl_tcp = get_calibrated_params()

    # 1 strip TCP offset
    T_base_fl_target = T_base_tcp_target @ np.linalg.inv(T_fl_tcp)
    p_target = T_base_fl_target[:3, 3]
    R_target = T_base_fl_target[:3, :3]

    # 2 compute wrist centre
    d6 = d[5]  # last link offset along Z₅; careful: use calibrated value
    p_wc = p_target - d6 * R_target[:, 2]   # -Z of flange

    # 3 analytic solve θ₁-θ₃ (details in docs/ik_algorithm.md)
    candidate_sets = _solve_first_three_joints(p_wc, a, d)

    # 4 for each set, solve θ₄-θ₆
    results: list[IKResult] = []
    for θ1, θ2, θ3 in candidate_sets:
        R0_3 = _fk_rot_0_3(θ1, θ2, θ3, a, d, alpha)   # cheap 3-link FK rotation
        R3_6 = R0_3.T @ R_target
        θ4, θ5, θ6 = _solve_spherical_wrist(R3_6)
        q = np.array([θ1, θ2, θ3, θ4, θ5, θ6], dtype=float)

        # 5 apply delta_theta (dt) and joint_direction to match FK convention
        dh_thetas = q + dt
        dh_thetas = joint_dirs * dh_thetas  # element-wise multiply

        # 6 validate by FK
        T_chk_fl = dry_parse.fk_to_flange(a, alpha, d, joint_dirs, dh_thetas)
        T_chk_tcp = T_chk_fl @ T_fl_tcp
        pos_err_mm = np.linalg.norm((T_chk_tcp[:3, 3] - T_base_tcp_target[:3, 3]) * 1e3)
        rot_err_rad = np.linalg.norm(
            (dry_parse.R_scipy.from_matrix(T_chk_tcp[:3, :3]).inv() *
             dry_parse.R_scipy.from_matrix(T_base_tcp_target[:3, :3])).as_rotvec()
        )

        within_limits = util.in_joint_limits(q)
        if pos_err_mm < tol_mm and rot_err_rad < tol_rad:
            results.append(IKResult(q_rad=q, within_joint_limits=within_limits,
                                    pose_error_mm=pos_err_mm))
            if return_first_valid:
                return results

    return results
```

*All helper routines (`_solve_first_three_joints`, `_solve_spherical_wrist`, etc.) live in the same file or `util.py` and are pure-Python, vectorised where convenient.*

---

### 6 Enumerate & score solutions

*Ranking policy* (adjust in `util.py`):

1.  **Reachability** – discard solutions violating link lengths.
2.  **Joint limits** – UR5 nominal limits ±360°; use your robot’s real range.
3.  **Proximity to current pose** – if your controller exposes the live joint state, compute Σ|Δθ| to pick the “closest” branch for smooth motion.
4.  **Pose accuracy** – lowest `pose_error_mm` trumps.

The `solve_ik` API can later accept a `current_joint_state` argument to enable #3 without changing callers.

---

### 7 Unit & integration tests

1.  **Golden pairs** – capture 20 – 30 joint → TCP samples from the real UR controller, store in `tests/fixtures/`.
2.  `test_ik_sanity.py`:

    ```python
    from ur5_kinematics.ik import solve_ik
    from ur5_kinematics.parameters import dry_parse

    for q_deg, tcp in golden_pairs:
        T_target = dry_parse.fk_to_flange(...)  # reproduce pose
        sols = solve_ik(T_target)
        assert any(np.allclose(s.q_rad, np.deg2rad(q_deg), atol=1e-3) for s in sols)
    ```
3.  **Round-trip** – pick random reachable TCP poses, run IK, feed each solution through FK, ensure position ≤ 0.5 mm & orientation ≤ 0.5 deg.

Execute with `uv pip install -e . && pytest -q`.

---

### 8 Validation against the UR controller

*Exactly the workflow you already use for FK*:

1.  **Capture** – on the teach pendant or via RTDE, read a target TCP pose *and* the joint angles the controller produced.
2.  **Run IK** – `python scripts/ik_demo.py --pose X,Y,Z,Rx,Ry,Rz`.
3.  **Check** – the script prints all candidate solutions ranked by error; the controller’s own joint vector should appear with `pose_error_mm ≈ 0`.
4.  **Tolerance** – aim for < 0.1 mm position & < 0.1 ° orientation; any residual error will largely stem from floating-point differences and ignore-singularity branches.

---

### 9 Future extensions (optional)

*   Numerical fallback (Levenberg-Marquardt) if analytic fails near singularities.
*   Differential IK (Jacobian pseudo-inverse) for velocity control.
*   Trajectory-level optimisation that blends branch switches gracefully.

---

### 10 Checklist before merging

*   [ ] Black + Ruff pass clean.
*   [ ] `pytest -q` passes on at least Python 3.10 and 3.12.
*   [ ] `mypy --strict` shows no new errors.
*   [ ] Docs updated (`docs/ik_algorithm.md`, `README.md` quick-start).
*   [ ] Verify `uv deploy` lock files include *no* extra heavyweight deps (SciPy you already have).
*   [ ] Tag release `v1.1.0` (semantic version: minor feature).

---
