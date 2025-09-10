import torch


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a 3D vector by the inverse of a quaternion.

    Args:
        q: The quaternion (w, x, y, z) with shape (4,).
        v: The vector (x, y, z) with shape (3,).

    Returns:
        The rotated vector (x, y, z) with shape (3,).
    """
    q_w = q[0]
    q_vec = q[1:]
    
    a = v * (2.0 * q_w**2 - 1.0)
    b = 2.0 * q_w * torch.linalg.cross(q_vec, v)
    c = 2.0 * q_vec * torch.dot(q_vec, v)
    
    return a - b + c




