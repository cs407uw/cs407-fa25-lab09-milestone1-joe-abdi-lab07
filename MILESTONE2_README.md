# Milestone 2: Sensor Data Processing for PDR

This directory contains the analysis script and dataset for Milestone 2 of Lab 9.

## Setup

### Prerequisites
You need Python 3.6+ with the following packages:
- numpy
- pandas
- matplotlib
- scipy

### Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas matplotlib scipy
```

## Dataset Files

The dataset files should be in `app/src/main/java/com/`:
- `ACCELERATION.csv` - For Part 1 (sensor error analysis)
- `WALKING.csv` - For Part 2 (step detection, should detect 37 steps)
- `TURNING.csv` - For Part 3 (direction detection, 90-degree turns)
- `WALKING_AND_TURNING.csv` - For Part 4 (trajectory plotting)

## Running the Analysis

Run the main analysis script:

```bash
python3 milestone2_analysis.py
```

This will generate:
- `part1_acceleration_analysis.png` - Acceleration, speed, and distance plots
- `part2_step_detection.png` - Step detection visualization
- `part3_turn_detection.png` - Turn detection visualization
- `part4_trajectory.png` - Trajectory plot

## Algorithm Details

### Part 1: Sensor Data Errors
- Integrates acceleration to get speed
- Integrates speed to get distance
- Compares results using actual vs noisy acceleration

### Part 2: Step Detection
- Uses vertical acceleration component (z-axis)
- Applies moving average smoothing
- Detects peaks using scipy's peak detection
- Expected: 37 steps in WALKING.csv

### Part 3: Direction Detection
- Uses gyroscope z-axis (rotation around vertical axis)
- Detects continuous turn regions
- Integrates angular velocity to estimate turn angles
- Expected: 8 turns (4 clockwise + 4 counter-clockwise) in TURNING.csv

### Part 4: Trajectory Plotting
- Combines step detection and turn detection
- Each step = 1 meter forward
- Applies turns to change direction
- Plots the resulting path

## Tuning Parameters

If the algorithms don't detect the expected number of steps/turns, you can adjust:

**Step Detection** (in `detect_steps` function):
- `threshold_factor`: Lower values detect more steps (default: 0.3)
- `min_step_interval`: Minimum time between steps in seconds (default: 0.4)
- `window_size`: Smoothing window size (default: 15)

**Turn Detection** (in `detect_turns` function):
- `threshold`: Minimum angular velocity for a turn (default: 0.3 rad/s)
- `min_turn_duration`: Minimum duration for a turn (default: 0.2 seconds)

## Output

The script prints:
- Part 1: Final distances and difference
- Part 2: Number of steps detected
- Part 3: Number of turns and their angles
- Part 4: Final position and total distance

All plots are saved as PNG files in the project root directory.

