"""Released endpoint replay models kept separate from causal equations."""

import numpy as np

from .records import Record


class ReleasedEndpointModel:
    """Replay a pinned released model endpoint on released observations."""

    # Released two-point files store separation first, then one column per
    # multipole; covariance keys are matched on rounded separations.
    _separation_column = 0
    _first_multipole_column = 1
    _separation_key_decimals = 6

    def _sdss_key(self, catalogue: str, multipole: int, separation: float) -> tuple[str, int, float]:
        return catalogue, int(multipole), round(
            float(separation), self._separation_key_decimals
        )

    def predict_sdss(self, inputs: Record) -> Record:
        results = []
        for item in inputs["tracers"]:
            observed_rows: list[float] = []
            predicted_rows: list[float] = []
            keys: list[tuple[str, int, float]] = []
            data = item["observed_table"]
            endpoint = item["released_endpoint_table"]
            column = self._separation_column
            for model_column, multipole in enumerate(
                item["multipoles"], start=self._first_multipole_column
            ):
                valid = (data[:, column] >= endpoint[:, column].min()) & (
                    data[:, column] <= endpoint[:, column].max()
                )
                separation = data[valid, column]
                observed_rows.extend(
                    data[valid, self._first_multipole_column + multipole]
                )
                predicted_rows.extend(
                    np.interp(
                        separation,
                        endpoint[:, column],
                        endpoint[:, model_column],
                    )
                )
                keys.extend(
                    self._sdss_key(item["catalogue"], multipole, value)
                    for value in separation
                )
            positions = {key: index for index, key in enumerate(keys)}
            covariance = np.full((len(keys), len(keys)), np.nan)
            rows = item["covariance"]
            for index, value in enumerate(rows["values"]):
                left = self._sdss_key(
                    rows["left_catalogue"][index],
                    rows["left_multipole"][index],
                    rows["left_separation"][index],
                )
                right = self._sdss_key(
                    rows["right_catalogue"][index],
                    rows["right_multipole"][index],
                    rows["right_separation"][index],
                )
                if left in positions and right in positions:
                    covariance[positions[left], positions[right]] = value
            if not np.all(np.isfinite(covariance)):
                raise ValueError(f"incomplete {item['tracer']} covariance")
            results.append(
                {
                    "tracer": item["tracer"],
                    "observed": np.asarray(observed_rows),
                    "predicted": np.asarray(predicted_rows),
                    "covariance": 0.5 * (covariance + covariance.T),
                }
            )
        return {"tracers": tuple(results), "source": inputs["source"]}
