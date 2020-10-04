import numpy as np


def approximation_line(points):
    """
    Решение задачи линейной апроксимации
    методом наименьших квадратов

    решение системы
    {a*sum(Xi^2) + b*sum(Xi) = sum(Xi*Yi)
    {a*sum(Xi) + b*n = sum(Yi)

    :param points: входящая матрица nympy, points[0] - Ox, points[1] - Oy
    :return:
    """
    n = points.shape[1]
    # реши через numpy
    m11 = np.sum(points[0]**2)
    m12 = np.sum(points[0])
    c1 = np.sum(np.prod(points, axis=0))
    c2 = np.sum(points[1])
    m = np.array([[m11, m12], [m12, n]])
    c = np.array([c1, c2])
    return np.linalg.solve(m, c)

points = np.array([[1, 2, 3],
                   [7, 7, 7]])
print(approximation_line(points))