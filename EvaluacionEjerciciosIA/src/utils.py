import numpy as np

def calculate_angle(p1, p2, p3):
    """Calcula el ángulo entre tres puntos en grados."""
    # Convierte a arrays numpy
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    # Vectores
    v1 = p1 - p2
    v2 = p3 - p2
    # Producto punto y magnitudes
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    # Evita división por cero
    if mag_v1 * mag_v2 == 0:
        return 0.0
    # Coseno del ángulo
    cos_theta = dot_product / (mag_v1 * mag_v2)
    # Corrige valores fuera de [-1, 1] por redondeo
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # Ángulo en grados
    angle = np.degrees(np.arccos(cos_theta))
    return angle