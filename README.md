# zermelo

## Overview

`zermelo.py` provides tools for calculating optimal navigation paths (Zermelo's problem) for underwater gliders or similar vehicles in the presence of ocean currents. It includes routines for projecting coordinates, loading velocity fields, simulating direct and optimal (Zermelo) trajectories, and visualizing results.

## Features

- **Spoint class**: Represents a point in space with position, heading, and history.
- **Zermelo solution**: Computes optimal paths considering current velocity fields.
- **Direct flight**: Simulates straight-line navigation ignoring currents.
- **Coordinate transformations**: Converts between geographic (lat/lon) and Gauss-Krueger projected coordinates.
- **Velocity field loading**: Loads and processes ocean current data from NetCDF files.
- **Visualization**: Plots velocity fields and computed trajectories.

## Main Functions

### `zermelo_solution(A, B, u, v, w, start_time, dt, dalpha, dl)`
Calculates the optimal path from point A to B considering the current field.

- **Inputs**:
  - `A`, `B`: Start and end coordinates `[lon, lat]`
  - `u`, `v`: x/y components of current velocity (xarray)
  - `w`: Vehicle speed (m/s)
  - `start_time`: Start time (datetime)
  - `dt`: Time step (seconds)
  - `dalpha`: Heading resolution (radians)
  - `dl`: Pruning distance (meters)
- **Outputs**: Path coordinates, headings, and times.

### `direct_flight(A, B, u, v, w, dt)`
Simulates a straight-line path from A to B, ignoring currents.

### `load_velocity_field(B, data_dir)`
Loads and processes ocean current data for the region around B.

### `plot_velocity_field(u, v, w)`
Visualizes the velocity field.

### `plot_solutions(ax, X, B, ...)`
Plots computed trajectories on a map.

### `get_waypoint(A, B, ...)`
Computes a waypoint between A and B using the Zermelo solution.

## Usage Example

```python
if __name__ == "__main__":
    B = np.array([-65.0, 19.5])  # End point
    A = np.array([-64.3, 18.5])  # Start point
    w = 0.25  # Speed (m/s)
    dt = 3600 * 6  # 6 hours
    dalpha = np.pi / 9
    dl = 0.3 * w * dt
    __main__(A, B, w, dt, dalpha, dl)
```

## Requirements

- Python 3.x
- numpy
- matplotlib
- xarray
- pyproj

## Notes

- Ocean current data should be available as a NetCDF file named `model_latest.nc` in the specified data directory.
- The module is designed for research and educational purposes.

## License

This project is licensed under the MIT License.
