import cv2
import numpy as np


class TextureDetector:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path

    def convert_to_grayscale(self) -> np.ndarray:
        image = cv2.imread(self.image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def calculate_glcm(
        self, image: np.ndarray, distance: int, angle: int, gray_levels=256
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray) or len(image.shape) != 2:
            raise ValueError("La imagen debe ser una matriz 2D")

        if not isinstance(distance, (int, np.integer)) or distance <= 0:
            raise ValueError("La distancia debe ser un entero positivo")

        if image.max() >= gray_levels or image.min() < 0:
            raise ValueError(
                f"Los valores de la imagen deben estar en el rango [0, {gray_levels-1}]"
            )

        # Calcular desplazamientos
        angle_radians = np.deg2rad(angle)
        dx = int(round(np.cos(angle_radians) * distance))
        dy = int(round(np.sin(angle_radians) * distance))

        # Obtener dimensiones
        rows, cols = image.shape

        # Inicializar GLCM
        glcm = np.zeros((gray_levels, gray_levels), dtype=np.uint32)

        # Crear índices
        i_indices = np.arange(rows)
        j_indices = np.arange(cols)

        # Crear meshgrid con el orden correcto
        ii, jj = np.meshgrid(i_indices, j_indices, indexing="ij")

        # Calcular coordenadas de vecinos
        ii_neighbors = ii + dy
        jj_neighbors = jj + dx

        # Crear máscara para índices válidos
        mask = (
            (ii_neighbors >= 0)
            & (ii_neighbors < rows)
            & (jj_neighbors >= 0)
            & (jj_neighbors < cols)
        )

        # Obtener índices válidos
        valid_indices = np.where(mask)

        # Obtener valores actuales y vecinos
        current_values = image[valid_indices]
        neighbor_values = image[
            ii_neighbors[valid_indices], jj_neighbors[valid_indices]
        ]

        # Actualizar GLCM
        np.add.at(glcm, (current_values, neighbor_values), 1)

        # Normalizar
        glcm = glcm.astype(np.float64)
        sum_value = glcm.sum()
        if sum_value > 0:
            glcm /= sum_value

        return glcm

    def extract_glcm_features(self, glcm: np.ndarray) -> dict:
        i, j = np.meshgrid(
            np.arange(glcm.shape[0]), np.arange(glcm.shape[1]), indexing="ij"
        )

        # Calcular características
        energy = np.sum(glcm ** 2)
        contrast = np.sum((i - j) ** 2 * glcm)
        epsilon = 1e-10
        entropy = -np.sum(glcm * np.log2(glcm + epsilon))
        homogeneity = np.sum(glcm / (1 + (i - j) ** 2))

        return {
            "contrast": contrast,
            "homogeneity": homogeneity,
            "energy": energy,
            "entropy": entropy,
        }


if __name__ == "__main__":
    # Crear instancia del detector
    td = TextureDetector("DS1/crosshatched_0033.jpg")

    # Convertir imagen a escala de grises
    image = td.convert_to_grayscale()

    # Calcular GLCM
    glcm = td.calculate_glcm(image, distance=1, angle=0)

    # Extraer características
    features = td.extract_glcm_features(glcm)

    # Mostrar resultados
    print("GLCM Matrix:")
    print(glcm)
    print("\nGLCM Features:")
    for feature, value in features.items():
        print(f"{feature}: {value:.4f}")
