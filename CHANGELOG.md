# Changelog

All notable changes to this project will be documented in this file.

## [0.0.1] - 2025-05-13

### Added
- Initial release of `dry_parse.py` providing calibration and forward kinematics tooling:
  - `parse_conf_list` for parsing list values from config files.
  - `load_calibration` to load delta calibration parameters from `calibration.conf`.
  - `load_urcontrol_config` to load UR control D-H parameters and TCP pose from `urcontrol.conf.UR5`.
  - `dh_matrix`, `fk_to_flange`, and `tcp_transform` functions for computing transformation matrices.
  - `compare_poses` for comparing calculated vs. measured TCP poses with error metrics.
  - CLI interface in `main` supporting `--fk` and `--compare` modes with input validation and formatted output.
  - Error handling for missing config files and invalid user inputs.
