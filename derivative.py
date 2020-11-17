import numpy as np
import numba


@numba.jit
def derivative(u, h, axis):
    if axis == 1:
        u_shift_down = np.roll(u, 1)
        u_shift_up = np.roll(u, -1)
    if axis == 0:
        u_transposed = np.transpose(u)

        u_shift_down = np.roll(u_transposed, 1)
        u_shift_down[:, 0] = np.roll(u_shift_down[:, 0], -1)
        u_shift_down = np.transpose(u_shift_down)

        u_shift_up = np.roll(u_transposed, -1)
        u_shift_up[:, 0] = np.roll(u_shift_up[:, 0], 1)
        u_shift_up = np.transpose(u_shift_up)

    du = (u_shift_up - u_shift_down) / (2 * (h[1] - h[0]))

    # # Indeces for some element along the axis
    # # du[:, 0] is the same as du[(range(du.shape[0]), 0)]
    # ind_first = tuple(0 if i == axis else np.arange(du.shape[i]) for i in np.arange(du.ndim))
    # ind_second = tuple(1 if i == axis else range(du.shape[i]) for i in range(du.ndim))
    # ind_prelast = tuple(-2 if i == axis else range(du.shape[i]) for i in range(du.ndim))
    # ind_last = tuple(-1 if i == axis else range(du.shape[i]) for i in range(du.ndim))
    #
    # du[ind_first] = (u[ind_second] - u[ind_first]) / (h[1] - h[0])
    # du[ind_last] = (u[ind_last] - u[ind_prelast]) / (h[1] - h[0])

    if axis == 1:
      du[:, 0] = (u[:, 1] - u[:, 0]) / (h[1] - h[0])
      du[:, -1] = (u[:, -1] - u[:, -2]) /  (h[1] - h[0])

    if axis == 0:
       du[0, :] = (u[1, :] - u[0, :]) /  (h[1] - h[0])
       du[-1, :] = (u[-1, :] - u[-2, :]) /  (h[1] - h[0])

    return du


def second_derivative(u, x, t, axis):
    if axis == 1:
        u_shift_down = np.roll(u, 1)
        u_shift_up = np.roll(u, -1)
        du = (u_shift_up - 2 * u + u_shift_down) / (x[0, 1] - x[0, 0]) ** 2
        du[:, 0] = (u[:, 2] - 2 * u[:, 1] + u[:, 0]) / (x[0, 1] - x[0, 0]) ** 2
        du[:, -1] = (u[:, -3] - 2 * u[:, -2] + u[:, -1]) / (x[0, 1] - x[0, 0]) ** 2

    if axis == 0:
        u_shift_down = np.roll(u, 1, axis=0)
        u_shift_up = np.roll(u, -1, axis=0)
        du = (u_shift_up - 2 * u + u_shift_down) / (t[1, 0] - t[0, 0]) ** 2
        du[0, :] = (u[2, :] - 2 * u[1, :] + u[0, :]) / (t[1, 0] - t[0, 0]) ** 2
        du[-1, :] = (u[-3, :] - 2 * u[-2, :] + u[-1, :]) / (t[1, 0] - t[0, 0]) ** 2

    return du


if __name__ == "__main__":
    x1 = np.linspace(0, 1, 10)
    t1 = np.linspace(0, 1, 10)

    x, t = np.meshgrid(x1, t1)
    u = x + t ** 2

    du = derivative(u, x1, axis=0)
    du = derivative(du, t1, axis=0)
    print(du)
