# UR_kinematics_calib

High-accuracy Python FK & IK solver for UR5 with calibrated DH parameters.

---

**Note on current IK Implementation:**
- The current inverse kinematics (IK) solver may converge at local minima depending on the initial guess provided. This is a known limitation of the present approach.
- **Future Plan:** There are plans to integrate a more robust IK solver to improve reliability and global convergence.
- **Analytical IK Research:** A [summary](docs/Executive Summary by Deep Research.pdf) on analytical IK using calibrated data has been added, based on Deep Research by ChatGPT.
---

## Project Structure

```
ur_kinematics_calib/
├── configs/             # calibration and UR control files
├── data/                # UR capture data (joint angles & EEF poses)
├── docs/                # documentation and plans
├── scripts/             # demo scripts
│   └── fk_demo.py       # forward kinematics demo
├── src/                 # package source
│   └── ur_kinematics_calib/
│       ├── util.py
│       ├── fk.py
│       └── ik.py
├── tests/               # test utilities
│   ├── dry_parse.py     # calibration & FK CLI
│   └── run_all_tests.py # automated test runner
└── README.md
```

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv):

> [!NOTE]  
> uv will create a venv and install the required dependencies with: `uv run ..`

## Usage

Run the FK demo with:
```bash
uv run scripts/fk_demo.py -j 1.54,-28.43,24.41,-130.54,-37.17,-147.01
```

Run the IK demo:
```bash
uv run scripts/ik_demo.py -p 159.13,-317.1,413.36,1.266,-3.32,-0.283
```

Run IK check with:
```bash
uv run scripts/fk_ik_check.py -j 31.64,-117.58,104.85,-77.19,-82.32,-58.4
```

> [!NOTE]  
> arguments:   
> `fk_demo.py` -j <deg1,deg2,deg3,deg4,deg5,deg6>  
> `fk_ik_check.py` -j <deg1,deg2,deg3,deg4,deg5,deg6>

Run the test with:

```bash
uv run tests/run_all_tests.py
```

Compare calculated FK result with actual pose:
```bash
uv run tests/dry_parse.py --compare 1.54,-28.43,24.41,-130.54,-37.17,-147.01,-872.69,-236.61,417.99,1.344,-1.557,0.494
```

### Result
- Pos error norm(mm): under 0.08 mm
    - minimum of 0.0169 mm to a maximum of 0.0800 mm.
- Rot error (deg): under 0.032 deg
    - minimum of 0.0119 deg to a maximum of 0.0317 deg.
