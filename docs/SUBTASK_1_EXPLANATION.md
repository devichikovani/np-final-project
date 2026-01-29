# Subtask 1: Grid → Handwritten Name

## Mathematical Analysis of Initial Formation Transition

---

# 1. OVERVIEW

**Goal:** Transition 2000 drones from a uniform grid to form a handwritten name.

**IVP Model (Static Target):**
```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T - x) + Frep - Kd*v]

Initial conditions:
x(0) = grid_position
v(0) = 0
```

---

# 2. INITIAL POSITIONS: UNIFORM GRID

## 2.1 Grid Generation Formula

```python
def generate_grid(n, width, height, margin=50):
    """Generate uniform grid of n points."""
    aspect = width / height
    rows = int(np.sqrt(n / aspect))
    cols = int(n / rows) + 1
    
    x_step = (width - 2*margin) / (cols - 1)
    y_step = (height - 2*margin) / (rows - 1)
    
    pts = []
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= n:
                break
            x = margin + j * x_step
            y = margin + i * y_step
            pts.append([x, y])
    
    return np.array(pts[:n])
```

## 2.2 Mathematical Representation

For a grid of n points in a W × H canvas:

```
Grid spacing: Δx = (W - 2m) / (cols - 1)
              Δy = (H - 2m) / (rows - 1)

Position: x[i,j] = m + j*Δx
          y[i,j] = m + i*Δy
```

where m = margin, cols = √(n × W/H), rows = n / cols.

---

# 3. TARGET EXTRACTION: HANDWRITTEN NAME

## 3.1 Edge Detection Pipeline

```
Input Image → Grayscale → Canny → Point Sampling
```

**Canny Edge Detection:**
```python
edges = canny_edge_detection(image, low=50, high=150)
edge_points = np.argwhere(edges > 0)  # All edge pixels
```

## 3.2 Point Sampling for n Drones

```python
def sample_edge_points(edges, n_drones):
    """Sample exactly n points from edge map."""
    points = np.argwhere(edges > 0)
    
    if len(points) > n_drones:
        indices = np.linspace(0, len(points)-1, n_drones).astype(int)
        points = points[indices]
    elif len(points) < n_drones:
        # Duplicate points to reach n_drones
        repeats = n_drones // len(points) + 1
        points = np.tile(points, (repeats, 1))[:n_drones]
    
    return points
```

---

# 4. ASSIGNMENT: GREEDY MATCHING

## 4.1 Problem Statement

Match each grid position to exactly one target point, minimizing total travel distance.

## 4.2 Distance Matrix

```
D[i,j] = ||grid[i] - target[j]||
```

For 2D:
```
D[i,j] = sqrt((gx[i] - tx[j])² + (gy[i] - ty[j])²)
```

## 4.3 Greedy Algorithm

```
For each drone i:
    Find nearest unassigned target j
    Assign target[j] to drone[i]
```

**Complexity:** O(n²) — acceptable for n = 2000.

---

# 5. THE IVP MODEL

## 5.1 State Variables

```
State vector: y = [x, v] where x, v ∈ ℝ²
```

For n = 2000 drones:
```
Full state: Y ∈ ℝ^(4000)  (2 position + 2 velocity per drone)
```

## 5.2 Governing Equations

```
dx/dt = v * min(1, Vmax/|v|)      # Position derivative (velocity-saturated)
dv/dt = (1/m)[Kp(T-x) + Frep - Kd*v]   # Velocity derivative (acceleration)
```

## 5.3 Force Analysis

### Spring Force (Attraction)
```
F_spring = Kp(T - x)
```
- Kp = 25.0 (spring constant)
- Acts toward target T
- Magnitude: |F_spring| = Kp × distance_to_target

### Damping Force (Energy Dissipation)
```
F_damp = -Kd × v
```
- Kd = 12.0 (damping coefficient)
- Opposes motion
- Critical for preventing oscillation

### Repulsion Force (Collision Avoidance)
```
F_rep = Σ Krep × (xi - xj) / |xi - xj|³   for all j where |xi-xj| < R_safe
```
- Krep = 50.0 (repulsion strength)
- R_safe = 4.0 pixels (minimum separation)

---

# 6. STABILITY ANALYSIS

## 6.1 Linearized System (Single Drone, No Repulsion)

For a single drone at position x with target at T = 0:
```
ẍ = (1/m)[-Kp*x - Kd*ẋ]
```

## 6.2 Characteristic Equation

```
m*s² + Kd*s + Kp = 0
```

**Roots:**
```
s = (-Kd ± √(Kd² - 4*m*Kp)) / (2*m)
```

With m = 1, Kd = 12, Kp = 25:
```
s = (-12 ± √(144 - 100)) / 2 = (-12 ± 6.6) / 2
s₁ = -2.7,  s₂ = -9.3
```

**Both eigenvalues are negative → System is stable.**

## 6.3 Damping Classification

```
Damping ratio: ζ = Kd / (2*√(m*Kp)) = 12 / (2*√25) = 1.2
```

Since ζ > 1: **Overdamped** — smooth approach without oscillation.

---

# 7. CONVERGENCE BEHAVIOR

## 7.1 Time Constant

For overdamped system:
```
τ = 1/|s₁| = 1/2.7 ≈ 0.37 seconds
```

Settling time (to within 2% of target):
```
t_settle ≈ 4τ ≈ 1.5 seconds
```

## 7.2 Trajectory Profile

The position follows:
```
x(t) = T + A*e^(s₁*t) + B*e^(s₂*t)
```

where A, B depend on initial conditions.

---

# 8. VELOCITY SATURATION

## 8.1 Why Saturate?

Physical drones have maximum speed limits. The model enforces:
```
v_effective = v × min(1, Vmax/|v|)
```

## 8.2 Effect on Dynamics

| Regime | Condition | Behavior |
|--------|-----------|----------|
| Normal | |v| < Vmax | Full dynamics |
| Saturated | |v| ≥ Vmax | Constant speed motion |

---

# 9. RK4 INTEGRATION

## 9.1 Why RK4?

Fourth-order accuracy O(h⁴) with single derivative evaluation per substep.

## 9.2 Per-Frame Steps

```python
STEPS_PER_FRAME = 10
DT = 0.05

for _ in range(STEPS_PER_FRAME):
    swarm.step()  # RK4 step with dt = 0.05
```

Total simulation time per frame: 10 × 0.05 = 0.5 seconds.

---

# 10. FRAME PARAMETERS

| Parameter | Value | Meaning |
|-----------|-------|---------|
| TRANSITION_FRAMES | 300 | Frames for grid→name |
| HOLD_FRAMES | 80 | Frames to hold name |
| STEPS_PER_FRAME | 10 | Physics steps per frame |
| FPS | 60 | Output video framerate |

**Transition duration:** 300 frames / 60 fps = 5 seconds

---

# 11. SUMMARY

**Subtask 1 solves the static-target IVP:**

```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T - x) + Frep - Kd*v]
```

**Key properties:**
- Overdamped (ζ = 1.2): No oscillation
- Stable: All eigenvalues negative
- Settling time: ~1.5 seconds

**No velocity matching needed** — targets are static.
