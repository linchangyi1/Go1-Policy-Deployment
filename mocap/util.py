import numpy as np
import math

def quaternion_to_euler(quaternion):
    x, y, z, w = quaternion
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def quat_to_rot_mat(quat):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Args:
        quat (numpy.ndarray): A numpy array of shape (4,) representing the quaternion [x, y, z, w].
    
    Returns:
        numpy.ndarray: A 3x3 rotation matrix.
    """
    quat = np.asarray(quat, dtype=np.float64)
    
    # Normalize the quaternion
    quat /= np.linalg.norm(quat)

    # Extract components
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # Precompute squared and cross terms
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Construct the rotation matrix
    R = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]
    ], dtype=np.float64)

    return R


def invert_transform(T):
    """Computes the inverse of a 4x4 transformation matrix."""
    R = T[0:3, 0:3]  # Extract rotation
    p = T[0:3, 3]    # Extract translation
    
    # Compute inverse
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = R.T  # Transpose of rotation matrix
    T_inv[0:3, 3] = -R.T @ p  # Negative translation rotated back
    
    return T_inv
    