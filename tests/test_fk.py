import numpy as np
import pytest
import quik_bind as quikpy
from ur_kinematics_calib.fk import dh_matrix, init_quik_robot, fk_to_flange, tcp_transform
from scipy.spatial.transform import Rotation as R_scipy

def test_dh_matrix_identity():
    T = dh_matrix(0, 0, 0, 0)
    expected = np.eye(4)
    assert np.allclose(T, expected)

def test_dh_matrix_rotation_z():
    theta = np.pi/2
    T = dh_matrix(0, 0, 0, theta)
    expected = np.array([[0, -1, 0, 0],
                         [1,  0, 0, 0],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1]])
    assert np.allclose(T, expected)

def test_dh_matrix_translation_z():
    d = 1.0
    T = dh_matrix(0, 0, d, 0)
    expected = np.eye(4)
    expected[2, 3] = d
    assert np.allclose(T, expected)

def test_dh_matrix_translation_x():
    a = 1.0
    T = dh_matrix(a, 0, 0, 0)
    expected = np.eye(4)
    expected[0, 3] = a
    assert np.allclose(T, expected)

def test_dh_matrix_rotation_x():
    alpha = np.pi/2
    T = dh_matrix(0, alpha, 0, 0)
    expected = np.array([[1,  0,  0, 0],
                         [0,  0, -1, 0],
                         [0,  1,  0, 0],
                         [0,  0,  0, 1]])
    assert np.allclose(T, expected)

def test_dh_matrix_combined_example():
    a = 1.0
    alpha = np.pi/2
    d = 1.0
    theta = np.pi/2
    T = dh_matrix(a, alpha, d, theta)
    expected = np.array([[0, 0, 1, 0],
                         [1, 0, 0, 1],
                         [0, 1, 0, 1],
                         [0, 0, 0, 1]])
    assert np.allclose(T, expected)

def test_fk_to_flange_python():
    a = np.zeros(6)
    alpha = np.zeros(6)
    d = np.zeros(6)
    thetas = np.zeros(6)
    T = fk_to_flange(a, alpha, d, None, thetas, use_quik=False)
    assert np.allclose(T, np.eye(4))

def test_fk_to_flange_quik():
    a = np.zeros(6)
    alpha = np.zeros(6)
    d = np.zeros(6)
    thetas = np.zeros(6)
    T = fk_to_flange(a, alpha, d, None, thetas, use_quik=True)
    assert np.allclose(T, np.eye(4))

def test_tcp_transform_translation():
    pose = np.array([1, 2, 3, 0, 0, 0])
    T = tcp_transform(pose)
    expected = np.eye(4)
    expected[0:3, 3] = [1, 2, 3]
    assert np.allclose(T, expected)

def test_tcp_transform_rotation():
    angle = np.pi / 2
    pose = np.array([0, 0, 0, angle, 0, 0])
    T = tcp_transform(pose)
    expected = np.eye(4)
    expected[0:3, 0:3] = R_scipy.from_rotvec([angle, 0, 0]).as_matrix()
    assert np.allclose(T, expected)

def test_init_quik_robot(monkeypatch):
    calls = {}
    def fake_init_robot(dh, link_types):
        calls['dh'] = dh.copy()
        calls['link_types'] = link_types.copy()
    monkeypatch.setattr(quikpy, 'init_robot', fake_init_robot)
    a = np.arange(6)
    alpha = np.arange(6) + 0.1
    d = np.arange(6) + 0.2
    init_quik_robot(a, alpha, d)
    assert 'dh' in calls
    assert np.allclose(calls['dh'][:, 0], a)
    assert np.allclose(calls['dh'][:, 1], alpha)
    assert np.allclose(calls['dh'][:, 2], d)
    assert np.all(calls['link_types'] == False) 