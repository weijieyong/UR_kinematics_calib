# calib_ik

High-accuracy Python FK & IK solver for UR5 with calibrated DH parameters.

## Project Structure

```
calib_ik/
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


## Usage

Setup venv and install the package with:

```bash
uv venv && uv pip install -e .
```

Run the FK demo with:
```bash
uv run scripts/fk_demo.py --joints 1.54,-28.43,24.41,-130.54,-37.17,-147.01
```

Compare calculated FK result with actual pose:
```bash
uv run tests/dry_parse.py --compare 1.54,-28.43,24.41,-130.54,-37.17,-147.01,-872.69,-236.61,417.99,1.344,-1.557,0.494
```

> [!NOTE]  
> fk_demo.py --joints <deg1,deg2,deg3,deg4,deg5,deg6>
> dry_parse.py --compare <j1,j2,j3,j4,j5,j6,Xmm,Ymm,Zmm,Rx,Ry,Rz>

Run the test with:

```bash
uv run tests/run_all_tests.py
```

### Result
- Pos error norm(mm): under 0.08 mm
    - minimum of 0.0169 mm to a maximum of 0.0800 mm.
- Rot error (deg): under 0.032 deg
    - minimum of 0.0119 deg to a maximum of 0.0317 deg.
