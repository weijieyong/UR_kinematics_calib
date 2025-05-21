import numpy as np
import pytest
from ur_kinematics_calib.util import parse_conf_list, load_calibration, load_urcontrol_config, compare_poses

def test_parse_conf_list_simple():
    assert parse_conf_list('[1, 2, 3]') == [1, 2, 3]

def test_parse_conf_list_with_comment():
    assert parse_conf_list('[1, 2, 3] # comment') == [1, 2, 3]

def test_load_calibration(tmp_path):
    config_file = tmp_path / 'calib.ini'
    content = '''
[mounting]
calibration_status = 2 Linearised
delta_theta = [1, 2, 3, 4, 5, 6]
delta_a = [6, 5, 4, 3, 2, 1]
delta_d = [0, 0, 0, 0, 0, 0]
delta_alpha = [0, 0, 0, 0, 0, 0]
'''
    config_file.write_text(content)
    delta_theta, delta_a, delta_d, delta_alpha = load_calibration(str(config_file))
    assert np.allclose(delta_theta, [1, 2, 3, 4, 5, 6])
    assert np.allclose(delta_a, [6, 5, 4, 3, 2, 1])
    assert np.allclose(delta_d, [0, 0, 0, 0, 0, 0])
    assert np.allclose(delta_alpha, [0, 0, 0, 0, 0, 0])

def test_load_urcontrol_config(tmp_path):
    config_file = tmp_path / 'urcontrol.ini'
    content = '''
[DH]
a = [1, 2, 3, 4, 5, 6]
d = [6, 5, 4, 3, 2, 1]
alpha = [0, 1, 2, 3, 4, 5]
q_home_offset = [0, 0, 0, 0, 0, 0]
joint_direction = [1, 1, 1, 1, 1, 1]

[Tool]
tcp_pose = [10, 20, 30, 0, 0, 0]
'''
    config_file.write_text(content)
    (a, d, alpha, q_home_offset, joint_direction), tcp_pose = load_urcontrol_config(str(config_file))
    assert np.allclose(a, [1, 2, 3, 4, 5, 6])
    assert np.allclose(d, [6, 5, 4, 3, 2, 1])
    assert np.allclose(alpha, [0, 1, 2, 3, 4, 5])
    assert np.allclose(q_home_offset, [0, 0, 0, 0, 0, 0])
    assert np.allclose(joint_direction, [1, 1, 1, 1, 1, 1])
    assert np.allclose(tcp_pose, [10, 20, 30, 0, 0, 0])

def test_compare_poses_identity(capsys):
    T = np.eye(4)
    measured = np.array([0, 0, 0, 0, 0, 0])
    compare_poses(T, measured)
    captured = capsys.readouterr()
    assert 'Pos Error' in captured.out
    assert 'Rot Error' in captured.out 