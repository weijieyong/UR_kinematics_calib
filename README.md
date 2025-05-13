# calib_ik

High-accuracy Python FK & IK solver for UR5 with calibrated DH parameters.

## Project Structure

```
calib_ik/
├── dry_parse.py       # main utility script
├── configs/          # calibration and UR control files
├── data/             # data captured from UR controller (each joint angles and the corresponding EEF pose)
├── docs/             # documentation and requirement specs
└── README.md
```


## Dependencies

Managed with [uv](https://github.com/astral-sh/uv):


## Usage

```bash
uv run dry_parse.py --joints_measured 20.72,-114.77,87.42,-62.33,-89.47,-68.88,-205.16,-220.39,628.03,0.018,3.137,-0.002
```
