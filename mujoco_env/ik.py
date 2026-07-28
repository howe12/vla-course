"""Minimal inverse kinematics using finite-difference gradient descent."""
import numpy as np

def solve_ik(env, joint_names_for_ik, body_name_trgt, q_init, p_trgt, R_trgt,
             max_ik_tick=100, ik_stepsize=1.0, ik_eps=1e-3, ik_th=np.radians(5.0),
             render=False, verbose_warning=True):
    """Solve IK using finite-difference gradient descent via MuJoCo."""
    ik_err_stack = []
    ik_info = "converged"

    q = np.array(q_init, dtype=np.float64)
    p_err_best = float("inf")
    q_best = q.copy()
    n_joints = len(joint_names_for_ik)
    eps = 1e-3

    for tick in range(max_ik_tick):
        env.forward(q=q, joint_names=joint_names_for_ik, increase_tick=False)
        p_curr, R_curr = env.get_pR_body(body_name=body_name_trgt)

        p_err = np.linalg.norm(p_trgt - p_curr)
        R_err = np.linalg.norm(R_trgt - R_curr)

        if p_err < ik_eps:
            ik_info = "converged"
            break

        if p_err < p_err_best:
            p_err_best = p_err
            q_best = q.copy()

        # Finite-difference gradient for position error
        grad = np.zeros(n_joints)
        for j_idx in range(n_joints):
            q_plus = q.copy()
            q_plus[j_idx] += eps
            env.forward(q=q_plus, joint_names=joint_names_for_ik, increase_tick=False)
            p_plus, _ = env.get_pR_body(body_name=body_name_trgt)
            e_plus = np.linalg.norm(p_trgt - p_plus)

            q_minus = q.copy()
            q_minus[j_idx] -= eps
            env.forward(q=q_minus, joint_names=joint_names_for_ik, increase_tick=False)
            p_minus, _ = env.get_pR_body(body_name=body_name_trgt)
            e_minus = np.linalg.norm(p_trgt - p_minus)

            grad[j_idx] = (e_minus - e_plus) / (2 * eps)

        # Gradient step
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            grad = grad / grad_norm
        q = q - ik_stepsize * grad

        ik_err_stack.append((p_err, R_err))

    else:
        ik_info = "max_iter"
        q = q_best

    return q, ik_err_stack, ik_info
