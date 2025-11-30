"""
Milestone 2: Sensor Data Processing for Pedestrian Dead-Reckoning (PDR)
This script implements:
- Part 1: Sensor Data Error Analysis
- Part 2: Step Detection
- Part 3: Direction Detection
- Part 4: Trajectory Plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

# ============================================================================
# Part 1: Understanding Sensor Data Errors
# ============================================================================

def part1_analyze_acceleration():
    """Analyze ACCELERATION.csv and calculate distance errors."""
    print("=" * 60)
    print("Part 1: Understanding Sensor Data Errors")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('app/src/main/java/com/ACCELERATION.csv')
    
    timestamps = df['timestamp'].values
    actual_acc = df['acceleration'].values
    noisy_acc = df['noisyacceleration'].values
    
    # Sampling rate is 0.1s
    dt = 0.1
    
    # Calculate speeds by integrating acceleration
    # v(t) = v0 + integral(a(t)dt)
    actual_speed = np.cumsum(actual_acc) * dt
    noisy_speed = np.cumsum(noisy_acc) * dt
    
    # Calculate distances by integrating speed
    # d(t) = d0 + integral(v(t)dt)
    actual_distance = np.cumsum(actual_speed) * dt
    noisy_distance = np.cumsum(noisy_speed) * dt
    
    # Final distances
    final_actual_distance = actual_distance[-1]
    final_noisy_distance = noisy_distance[-1]
    difference = abs(final_actual_distance - final_noisy_distance)
    
    print(f"\nFinal distance (actual acceleration): {final_actual_distance:.4f} m")
    print(f"Final distance (noisy acceleration): {final_noisy_distance:.4f} m")
    print(f"Difference: {difference:.4f} m")
    print(f"Percentage error: {(difference/final_actual_distance)*100:.2f}%")
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Acceleration comparison
    axes[0].plot(timestamps, actual_acc, label='Actual Acceleration', linewidth=2)
    axes[0].plot(timestamps, noisy_acc, label='Noisy Acceleration', alpha=0.7, linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Acceleration: Actual vs Noisy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Speed comparison
    axes[1].plot(timestamps, actual_speed, label='Speed (Actual)', linewidth=2)
    axes[1].plot(timestamps, noisy_speed, label='Speed (Noisy)', alpha=0.7, linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Speed (m/s)')
    axes[1].set_title('Speed: Actual vs Noisy Acceleration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Distance comparison
    axes[2].plot(timestamps, actual_distance, label='Distance (Actual)', linewidth=2)
    axes[2].plot(timestamps, noisy_distance, label='Distance (Noisy)', alpha=0.7, linewidth=2)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Distance (m)')
    axes[2].set_title('Distance Traveled: Actual vs Noisy Acceleration')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part1_acceleration_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'part1_acceleration_analysis.png'")
    plt.close()
    
    return final_actual_distance, final_noisy_distance, difference

# ============================================================================
# Part 2: Step Detection
# ============================================================================

def smooth_data(data, window_size=5):
    """Apply moving average smoothing."""
    return uniform_filter1d(data, size=window_size, mode='nearest')

def detect_steps(accel_data, timestamps, threshold_factor=0.3, min_step_interval=0.4):
    """
    Detect steps using peak detection on the vertical acceleration component.
    For walking with phone face-up, the z-axis (vertical) shows the most variation.
    
    Args:
        accel_data: Array of shape (n, 3) with x, y, z accelerations
        timestamps: Array of timestamps
        threshold_factor: Factor to multiply std dev for threshold
        min_step_interval: Minimum time between steps (seconds)
    
    Returns:
        step_indices: Indices where steps were detected
        smoothed: Smoothed acceleration data
    """
    # Use vertical acceleration (z-axis) - this is most relevant for walking
    # When phone is face-up, z-axis points up and shows vertical motion
    vertical_accel = accel_data[:, 2]
    
    # Remove static gravity component (approximately 9.8 m/s²)
    vertical_accel = vertical_accel - np.mean(vertical_accel)
    
    # Smooth the data to reduce noise
    smoothed = smooth_data(vertical_accel, window_size=15)
    
    # Find peaks (steps typically show as peaks when foot hits ground)
    # Use a threshold based on the standard deviation
    mean_val = np.mean(smoothed)
    std_val = np.std(smoothed)
    threshold = mean_val + threshold_factor * std_val
    
    # Find peaks with minimum distance between them
    # Convert min_step_interval to samples (assuming ~5ms sampling rate)
    dt = (timestamps[1] - timestamps[0]) / 1e9  # Convert nanoseconds to seconds
    min_distance_samples = max(int(min_step_interval / dt), 10)
    
    # Find peaks - steps are typically positive peaks in vertical acceleration
    peaks, properties = signal.find_peaks(smoothed, 
                                         height=threshold,
                                         distance=min_distance_samples,
                                         prominence=std_val * 0.2)
    
    return peaks, smoothed

def part2_step_detection():
    """Detect steps in WALKING.csv."""
    print("\n" + "=" * 60)
    print("Part 2: Step Detection")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('app/src/main/java/com/WALKING.csv')
    
    timestamps = df['timestamp'].values
    accel_x = df['accel_x'].values
    accel_y = df['accel_y'].values
    accel_z = df['accel_z'].values
    accel_data = np.column_stack([accel_x, accel_y, accel_z])
    
    # Detect steps
    step_indices, smoothed_magnitude = detect_steps(accel_data, timestamps, 
                                                    threshold_factor=0.3, 
                                                    min_step_interval=0.4)
    
    num_steps = len(step_indices)
    print(f"\nNumber of steps detected: {num_steps}")
    print(f"Expected: 37 steps")
    
    # Convert timestamps to seconds for plotting
    time_seconds = (timestamps - timestamps[0]) / 1e9
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Acceleration magnitude with detected steps
    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2) - 9.8
    axes[0].plot(time_seconds, accel_magnitude, label='Raw Acceleration Magnitude', alpha=0.5, linewidth=1)
    axes[0].plot(time_seconds, smoothed_magnitude, label='Smoothed', linewidth=2)
    axes[0].scatter(time_seconds[step_indices], smoothed_magnitude[step_indices], 
                   color='red', s=100, zorder=5, label=f'Detected Steps ({num_steps})')
    axes[0].axhline(y=np.mean(smoothed_magnitude) + 0.3 * np.std(smoothed_magnitude), 
                   color='orange', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Acceleration Magnitude (m/s²)')
    axes[0].set_title('Step Detection in WALKING.csv')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Individual acceleration components
    axes[1].plot(time_seconds, accel_x, label='X', alpha=0.7, linewidth=1)
    axes[1].plot(time_seconds, accel_y, label='Y', alpha=0.7, linewidth=1)
    axes[1].plot(time_seconds, accel_z, label='Z', alpha=0.7, linewidth=1)
    axes[1].scatter(time_seconds[step_indices], accel_x[step_indices], 
                   color='red', s=50, zorder=5, marker='x')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('Acceleration Components with Step Markers')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part2_step_detection.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'part2_step_detection.png'")
    plt.close()
    
    return step_indices, num_steps

# ============================================================================
# Part 3: Direction Detection
# ============================================================================

def detect_turns(gyro_z, timestamps, threshold=0.5, min_turn_duration=0.1):
    """
    Detect 90-degree turns using gyroscope z-axis data.
    
    Args:
        gyro_z: Gyroscope z-axis data (rotation around vertical axis)
        timestamps: Array of timestamps
        threshold: Minimum angular velocity to consider a turn
        min_turn_duration: Minimum duration for a turn (seconds)
    
    Returns:
        turn_indices: Indices where turns start
        turn_angles: Estimated turn angles (degrees)
    """
    # Smooth the gyroscope data
    smoothed_gyro = smooth_data(gyro_z, window_size=10)
    
    # Convert timestamps to seconds
    dt = (timestamps[1] - timestamps[0]) / 1e9
    
    # Find regions where angular velocity exceeds threshold
    turn_mask = np.abs(smoothed_gyro) > threshold
    
    # Find continuous turn regions
    turn_regions = []
    in_turn = False
    turn_start = 0
    
    for i in range(len(turn_mask)):
        if turn_mask[i] and not in_turn:
            turn_start = i
            in_turn = True
        elif not turn_mask[i] and in_turn:
            # End of turn region
            turn_duration = (i - turn_start) * dt
            if turn_duration >= min_turn_duration:
                turn_regions.append((turn_start, i))
            in_turn = False
    
    # Calculate turn angles by integrating angular velocity
    turn_indices = []
    turn_angles = []
    
    for start, end in turn_regions:
        # Integrate angular velocity to get angle
        angle_rad = np.trapz(smoothed_gyro[start:end]) * dt
        angle_deg = np.degrees(angle_rad)
        
        # Only consider significant turns (close to 90 degrees)
        if abs(angle_deg) > 45:  # Allow some tolerance
            turn_indices.append(start)
            turn_angles.append(angle_deg)
    
    return turn_indices, turn_angles, smoothed_gyro

def part3_direction_detection():
    """Detect 90-degree turns in TURNING.csv."""
    print("\n" + "=" * 60)
    print("Part 3: Direction Detection")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('app/src/main/java/com/TURNING.csv')
    
    timestamps = df['timestamp'].values
    gyro_z = df['gyro_z'].values
    
    # Detect turns
    turn_indices, turn_angles, smoothed_gyro = detect_turns(gyro_z, timestamps, 
                                                            threshold=0.3,
                                                            min_turn_duration=0.2)
    
    num_turns = len(turn_indices)
    print(f"\nNumber of turns detected: {num_turns}")
    print(f"Expected: 8 turns (4 clockwise + 4 counter-clockwise)")
    
    if num_turns > 0:
        print("\nTurn angles (degrees):")
        for i, angle in enumerate(turn_angles):
            direction = "clockwise" if angle > 0 else "counter-clockwise"
            print(f"  Turn {i+1}: {angle:.2f}° ({direction})")
    
    # Convert timestamps to seconds
    time_seconds = (timestamps - timestamps[0]) / 1e9
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Gyroscope z-axis with detected turns
    axes[0].plot(time_seconds, gyro_z, label='Raw Gyroscope Z', alpha=0.5, linewidth=1)
    axes[0].plot(time_seconds, smoothed_gyro, label='Smoothed', linewidth=2)
    axes[0].scatter(time_seconds[turn_indices], smoothed_gyro[turn_indices], 
                   color='red', s=100, zorder=5, label=f'Detected Turns ({num_turns})')
    axes[0].axhline(y=0.3, color='orange', linestyle='--', label='Threshold')
    axes[0].axhline(y=-0.3, color='orange', linestyle='--')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Angular Velocity (rad/s)')
    axes[0].set_title('Turn Detection in TURNING.csv')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative angle (integrated angular velocity)
    cumulative_angle = np.cumsum(smoothed_gyro) * (time_seconds[1] - time_seconds[0])
    cumulative_angle_deg = np.degrees(cumulative_angle)
    axes[1].plot(time_seconds, cumulative_angle_deg, linewidth=2)
    axes[1].scatter(time_seconds[turn_indices], cumulative_angle_deg[turn_indices], 
                   color='red', s=100, zorder=5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Cumulative Angle (degrees)')
    axes[1].set_title('Cumulative Rotation Angle')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part3_turn_detection.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'part3_turn_detection.png'")
    plt.close()
    
    return turn_indices, turn_angles

# ============================================================================
# Part 4: Trajectory Plotting
# ============================================================================

def part4_trajectory_plotting():
    """Plot trajectory from WALKING_AND_TURNING.csv."""
    print("\n" + "=" * 60)
    print("Part 4: Trajectory Plotting")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('app/src/main/java/com/WALKING_AND_TURNING.csv')
    
    timestamps = df['timestamp'].values
    accel_x = df['accel_x'].values
    accel_y = df['accel_y'].values
    accel_z = df['accel_z'].values
    gyro_z = df['gyro_z'].values
    accel_data = np.column_stack([accel_x, accel_y, accel_z])
    
    # Detect steps
    step_indices, _ = detect_steps(accel_data, timestamps, 
                                   threshold_factor=0.3, 
                                   min_step_interval=0.4)
    
    # Detect turns
    turn_indices, turn_angles, _ = detect_turns(gyro_z, timestamps, 
                                                threshold=0.3,
                                                min_turn_duration=0.2)
    
    print(f"\nSteps detected: {len(step_indices)}")
    print(f"Turns detected: {len(turn_indices)}")
    
    # Build trajectory
    # Each step is approximately 1m in length
    step_length = 1.0  # meters
    
    # Starting position and direction
    x, y = 0.0, 0.0
    direction = 0.0  # degrees, 0 = North (positive y)
    
    trajectory_x = [x]
    trajectory_y = [y]
    
    # Convert timestamps to seconds
    time_seconds = (timestamps - timestamps[0]) / 1e9
    
    # Process steps and turns together
    all_events = []
    for idx in step_indices:
        all_events.append(('step', idx, time_seconds[idx]))
    for idx in turn_indices:
        all_events.append(('turn', idx, time_seconds[idx]))
    
    # Sort by time
    all_events.sort(key=lambda x: x[2])
    
    # Apply turns and steps in chronological order
    for event_type, idx, time in all_events:
        if event_type == 'turn':
            # Find the corresponding turn angle
            if idx in turn_indices:
                turn_idx = turn_indices.index(idx)
                angle = turn_angles[turn_idx]
                
                # Round to nearest multiple of 45 degrees (as per requirements)
                angle_rounded = round(angle / 45.0) * 45.0
                
                direction += angle_rounded
                direction = direction % 360  # Normalize to 0-360
        elif event_type == 'step':
            # Move forward in current direction
            direction_rad = np.radians(direction)
            x += step_length * np.sin(direction_rad)
            y += step_length * np.cos(direction_rad)
            trajectory_x.append(x)
            trajectory_y.append(y)
    
    print(f"\nFinal position: ({x:.2f}, {y:.2f})")
    print(f"Total distance traveled: {len(step_indices) * step_length:.2f} m")
    
    # Plot trajectory
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Trajectory
    axes[0].plot(trajectory_x, trajectory_y, 'b-', linewidth=2, label='Path')
    axes[0].scatter([0], [0], color='green', s=200, marker='o', 
                   label='Start', zorder=5)
    axes[0].scatter([x], [y], color='red', s=200, marker='s', 
                   label='End', zorder=5)
    axes[0].scatter(trajectory_x[1:], trajectory_y[1:], color='blue', 
                   s=30, alpha=0.5, zorder=4)
    axes[0].set_xlabel('X Position (m)')
    axes[0].set_ylabel('Y Position (m)')
    axes[0].set_title('Walking Trajectory from WALKING_AND_TURNING.csv')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Plot 2: Sensor data with events marked
    time_seconds = (timestamps - timestamps[0]) / 1e9
    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2) - 9.8
    
    axes[1].plot(time_seconds, accel_magnitude, label='Acceleration Magnitude', 
                alpha=0.5, linewidth=1)
    axes[1].scatter(time_seconds[step_indices], accel_magnitude[step_indices], 
                   color='green', s=50, label='Steps', zorder=5, marker='x')
    axes[1].scatter(time_seconds[turn_indices], 
                   [np.max(accel_magnitude) * 0.8] * len(turn_indices), 
                   color='red', s=100, label='Turns', zorder=5, marker='^')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Acceleration Magnitude (m/s²)')
    axes[1].set_title('Sensor Data with Detected Events')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part4_trajectory.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'part4_trajectory.png'")
    plt.close()
    
    return trajectory_x, trajectory_y

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Milestone 2: Sensor Data Processing for PDR")
    print("=" * 60)
    
    # Part 1
    try:
        part1_analyze_acceleration()
    except Exception as e:
        print(f"Error in Part 1: {e}")
    
    # Part 2
    try:
        part2_step_detection()
    except Exception as e:
        print(f"Error in Part 2: {e}")
    
    # Part 3
    try:
        part3_direction_detection()
    except Exception as e:
        print(f"Error in Part 3: {e}")
    
    # Part 4
    try:
        part4_trajectory_plotting()
    except Exception as e:
        print(f"Error in Part 4: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the generated PNG files for plots.")
    print("=" * 60)

