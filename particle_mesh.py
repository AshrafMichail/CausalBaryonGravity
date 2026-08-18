"""Shared particle-mesh machinery and the reference rerun."""

import numpy as np

from .records import Record


class ParticleMeshEngine:
    """Canonical one-dimensional CIC force and leapfrog implementation."""

    def cic_stencil(self, points: np.ndarray, cells: int, length: float) -> tuple[np.ndarray, np.ndarray]:
        coordinate = points * cells / length
        left = np.floor(coordinate).astype(int)
        return left, coordinate - left

    def cic_deposit(
        self, positions: np.ndarray, masses: np.ndarray, cells: int, length: float
    ) -> np.ndarray:
        density = np.zeros(cells)
        for component in range(len(positions)):
            left, right_weight = self.cic_stencil(positions[component], cells, length)
            np.add.at(density, left % cells, masses[component] * (1.0 - right_weight))
            np.add.at(density, (left + 1) % cells, masses[component] * right_weight)
        return density / np.mean(density) - 1.0

    def solve_force(self, density: np.ndarray, cells: int, length: float) -> np.ndarray:
        wave = np.fft.fftfreq(cells, d=length / cells) * 2.0 * np.pi
        transformed = np.fft.fft(density)
        force = np.zeros(cells, dtype=complex)
        nonzero = wave != 0.0
        force[nonzero] = 1j * transformed[nonzero] / wave[nonzero]
        return np.fft.ifft(force).real

    def cic_gather(
        self, force: np.ndarray, points: np.ndarray, cells: int, length: float
    ) -> np.ndarray:
        left, right_weight = self.cic_stencil(points, cells, length)
        return (
            force[left % cells] * (1.0 - right_weight)
            + force[(left + 1) % cells] * right_weight
        )

    def leapfrog(
        self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray,
        inputs: Record,
    ) -> np.ndarray:
        cells, length = inputs["cells"], inputs["length"]
        time_step = inputs["time_step"]
        for _ in range(inputs["steps"]):
            force = self.solve_force(
                self.cic_deposit(positions, masses, cells, length), cells, length
            )
            for component in range(len(positions)):
                velocities[component] += (
                    0.5 * time_step
                    * self.cic_gather(force, positions[component], cells, length)
                )
                positions[component] = np.mod(
                    positions[component] + time_step * velocities[component], length
                )
            force = self.solve_force(
                self.cic_deposit(positions, masses, cells, length), cells, length
            )
            for component in range(len(positions)):
                velocities[component] += (
                    0.5 * time_step
                    * self.cic_gather(force, positions[component], cells, length)
                )
        return self.cic_deposit(positions, masses, cells, length)


class ReferenceParticleMeshModel:
    """Repeat the initialized two-component reference evolution."""

    def __init__(self) -> None:
        self.engine = ParticleMeshEngine()

    def predict_density(self, inputs: Record, response_to_baryon_ratio: float) -> np.ndarray:
        positions = np.vstack([inputs["positions"].copy(), inputs["positions"].copy()])
        velocities = np.vstack([inputs["velocities"].copy(), inputs["velocities"].copy()])
        masses = np.asarray([1.0, response_to_baryon_ratio])[:, None]
        return self.engine.leapfrog(positions, velocities, masses, inputs)
