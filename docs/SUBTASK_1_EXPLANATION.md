# Subtask 1: Initial Grid → Handwritten Name

## Complete Theoretical and Implementation Explanation

---

# 1. OVERVIEW

## What This Subtask Does

**Objective:** Transform a swarm of drones from a regular grid formation into the shape of your handwritten name.

**Visual Sequence:**
```
Frame 0:                    Frame 75:                   Frame 150:
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ · · · · · · · · │        │    ·  · ·       │        │ ·····     ····  │
│ · · · · · · · · │   →    │  ·  · ·  ·      │   →    │ ·   ·    ·   ·  │
│ · · · · · · · · │        │ · ·  ··   ·     │        │ ·····    ·····  │
│ · · · · · · · · │        │   ·   · ·       │        │ ·  ·     ·   ·  │
│ · · · · · · · · │        │  ·  ·   ·       │        │ ·   ·    ·   ·  │
└─────────────────┘        └─────────────────┘        └─────────────────┘
   Initial Grid              Mid-Transition              "RAMAZ" Complete
```

---

# 2. MATHEMATICAL PROBLEM FORMULATION

## 2.1 Initial Condition

The simulation begins with drones arranged in a **uniform grid**:

$$\vec{x}_i(0) = \text{grid position}_i, \quad \vec{v}_i(0) = \vec{0}$$

All drones start at rest (zero velocity).

## 2.2 Target Configuration

The target shape is extracted from a handwritten name image:

$$\vec{T}_i = \text{assigned target position for drone } i$$

Each drone is assigned to exactly one target point on the name outline.

## 2.3 The Initial Value Problem (IVP)

We solve the following system of differential equations:

**Position Equation:**
$$\frac{d\vec{x}_i}{dt} = \vec{v}_i \cdot \min\left(1, \frac{v_{max}}{|\vec{v}_i|}\right)$$

**Velocity Equation:**
$$\frac{d\vec{v}_i}{dt} = \frac{1}{m}\left[k_p(\vec{T}_i - \vec{x}_i) + \vec{F}_{rep,i} - k_d\vec{v}_i\right]$$

**Initial Conditions:**
$$\vec{x}_i(0) = \text{grid}_i, \quad \vec{v}_i(0) = \vec{0}$$

---

# 3. PHYSICAL INTERPRETATION

## 3.1 Spring-Mass-Damper System

Each drone behaves like a mass attached to its target by a spring, with friction:

```
                    Spring (kₚ)
    [Target T] ----/\/\/\/\---- [Drone] ----→ Velocity v
                                   ↓
                              Damping (kd)
                              Friction force opposing motion
```

### Force Components:

| Force | Formula | Physical Meaning |
|-------|---------|------------------|
| **Spring Force** | $k_p(\vec{T} - \vec{x})$ | Pulls drone toward target. Stronger when far away. |
| **Damping Force** | $-k_d\vec{v}$ | Friction opposing motion. Prevents oscillation. |
| **Repulsion Force** | $\vec{F}_{rep}$ | Pushes away from nearby drones. Collision avoidance. |

## 3.2 Why This Model?

**Spring Force (Proportional Control):**
- Creates a restoring force toward the target
- Magnitude proportional to distance: far drones feel strong pull
- At target, force becomes zero (equilibrium)

**Damping (Derivative Control):**
- Opposes motion regardless of position
- Prevents overshooting and oscillation
- Creates smooth, stable convergence

**Together:** This is a PD (Proportional-Derivative) controller, widely used in robotics.

---

# 4. CODE IMPLEMENTATION

## 4.1 Grid Initialization

**Location:** `Swarm.__init__()` (lines 493-517)

```python
def __init__(self, n):
    self.n = n
    self.pos = np.zeros((n, 2))  # Position array
    self.vel = np.zeros((n, 2))  # Velocity array (starts at zero)
    self.tgt = np.zeros((n, 2))  # Target array
    
    # Create grid matching canvas aspect ratio
    cols = int(np.sqrt(n * W / H))
    rows = int(np.ceil(n / cols))
    xx, yy = np.meshgrid(
        np.linspace(80, W-80, cols),  # X coordinates
        np.linspace(80, H-80, rows)   # Y coordinates
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])[:n]
    self.pos = grid.copy()
```

### Grid Calculation Explained:

For 1200 drones on 800×600 canvas:

1. **Aspect ratio:** $\frac{W}{H} = \frac{800}{600} = 1.33$

2. **Columns:** $\sqrt{n \cdot \frac{W}{H}} = \sqrt{1200 \times 1.33} \approx 40$

3. **Rows:** $\lceil \frac{n}{cols} \rceil = \lceil \frac{1200}{40} \rceil = 30$

4. **Spacing:** 
   - X: $\frac{800 - 160}{39} \approx 16.4$ pixels between drones
   - Y: $\frac{600 - 160}{29} \approx 15.2$ pixels between drones

**Video Effect:** Drones appear in a neat rectangular grid at the start of the video.

---

## 4.2 Target Extraction from Image

**Location:** `extract_points()` (lines 430-477)

```python
def extract_points(source, n, is_text=False):
    # Load and process image
    img = cv2.imread(source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find outline
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Get edge pixel coordinates
    pts = np.column_stack(np.where(edges > 0))
    pts = np.column_stack([pts[:, 1], pts[:, 0]])  # Swap to (x, y)
    
    # Sample n points uniformly
    indices = np.linspace(0, len(pts) - 1, n, dtype=int)
    sampled = pts[indices]
    
    # Scale to canvas
    scale = min((W - 100) / w, (H - 100) / h)
    return scaled_points
```

### Edge Detection Purpose:

1. **Input:** Grayscale image of handwritten name
2. **Canny Edge Detection:** Finds pixels where intensity changes rapidly (edges)
3. **Output:** List of (x, y) coordinates on the name outline

**Why edges?** We want drones to form the *outline* of letters, not fill them solid.

---

## 4.3 Drone-to-Target Assignment

**Location:** `greedy_assignment()` (lines 131-154)

```python
def greedy_assignment(positions, targets):
    n = len(positions)
    dist = euclidean_distance_matrix(positions, targets)
    
    result = np.zeros((n, 2))
    assigned_targets = np.zeros(n, dtype=bool)
    
    for i in range(n):
        # Mask assigned targets with infinity
        masked_dist = dist[i].copy()
        masked_dist[assigned_targets] = np.inf
        
        # Find nearest available target
        j = np.argmin(masked_dist)
        result[i] = targets[j]
        assigned_targets[j] = True
    
    return result
```

### Algorithm:

```
For each drone i (in order):
    1. Compute distance to all unassigned targets
    2. Select nearest unassigned target
    3. Assign drone i → target j
    4. Mark target j as taken
```

### Complexity: O(n²)

**Why greedy instead of optimal (Hungarian)?**
- Hungarian algorithm: O(n³) - too slow for 1200 drones
- Greedy: O(n²) - fast enough, gives reasonable (not optimal) paths

**Video Effect:** Drones generally take short, direct paths. Some crossing may occur but it's acceptable.

---

## 4.4 Simulation Loop

**Location:** `Swarm.simulate()` (lines 647-660)

```python
def simulate(self, targets, frames):
    """Simulate transition to static targets."""
    # Assign each drone to a target
    self.tgt = greedy_assignment(self.pos, targets)
    
    result = []
    # Run physics simulation
    for _ in range(frames):
        self.step()  # RK4 integration
        result.append(self.pos.copy())
    
    # Hold at final position
    for _ in range(HOLD_FRAMES):
        self.step()
        result.append(self.pos.copy())
    
    return result
```

### Execution for Subtask 1:

```python
# In main():
t1 = extract_points("handwritten_name.png", n)  # Get target points
phase1_frames = swarm.simulate(t1, TRANSITION_FRAMES)  # Run simulation
```

1. Extract 1200 points from handwritten name image
2. Assign each drone to a target point
3. Run RK4 simulation for 150 frames
4. Hold position for 40 frames
5. Return all 190 position snapshots

---

# 5. NUMERICAL METHOD: RK4

## 5.1 Why RK4?

The differential equations can't be solved analytically because:
- Repulsion forces depend on all drone positions
- Velocity saturation is non-linear
- Multiple coupled equations

**Numerical solution:** Approximate the solution by stepping forward in small time increments.

## 5.2 RK4 vs Simpler Methods

| Method | Formula | Error per Step | For DT=0.05 |
|--------|---------|----------------|-------------|
| Euler | $y_{n+1} = y_n + h \cdot f(y_n)$ | O(h²) = 0.0025 | ~2.5% error |
| RK2 | Uses 2 evaluations | O(h³) = 0.000125 | ~0.01% error |
| **RK4** | Uses 4 evaluations | O(h⁵) = 0.0000003 | ~0.00003% error |

RK4 is **10,000× more accurate** than Euler for the same step size.

## 5.3 RK4 Implementation

**Location:** `Swarm.step()` (lines 601-645)

```python
def step(self):
    def deriv(p, v):
        # Velocity saturation
        vnorm = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
        dx = v * np.minimum(1.0, V_MAX / vnorm)
        
        # Acceleration
        rep = self.repulsion()  # Collision avoidance
        dv = (K_P * (self.tgt - p) + rep - K_D * v) / M
        return dx, dv
    
    # Four stages of RK4
    k1x, k1v = deriv(self.pos, self.vel)
    k2x, k2v = deriv(self.pos + 0.5*DT*k1x, self.vel + 0.5*DT*k1v)
    k3x, k3v = deriv(self.pos + 0.5*DT*k2x, self.vel + 0.5*DT*k2v)
    k4x, k4v = deriv(self.pos + DT*k3x, self.vel + DT*k3v)
    
    # Weighted average update
    self.pos += (DT/6) * (k1x + 2*k2x + 2*k3x + k4x)
    self.vel += (DT/6) * (k1v + 2*k2v + 2*k3v + k4v)
```

---

# 6. CONVERGENCE ANALYSIS

## 6.1 Will Drones Reach Their Targets?

**Linearized system (ignoring repulsion and saturation):**

$$\frac{d^2\vec{x}}{dt^2} + \frac{k_d}{m}\frac{d\vec{x}}{dt} + \frac{k_p}{m}(\vec{x} - \vec{T}) = 0$$

This is a **damped harmonic oscillator** with characteristic equation:

$$\lambda^2 + \frac{k_d}{m}\lambda + \frac{k_p}{m} = 0$$

**Solutions:**
$$\lambda = \frac{-k_d \pm \sqrt{k_d^2 - 4mk_p}}{2m}$$

## 6.2 Damping Ratio

$$\zeta = \frac{k_d}{2\sqrt{mk_p}}$$

With our parameters: $m=1$, $k_p=25$, $k_d=12$:

$$\zeta = \frac{12}{2\sqrt{1 \times 25}} = \frac{12}{10} = 1.2$$

**Since ζ > 1:** The system is **overdamped**.

### What This Means:
- Drones approach targets **without oscillation**
- No bouncing back and forth
- Smooth, exponential convergence

## 6.3 Time to Reach Target

For overdamped system, the slower decay rate is:

$$\lambda_{slow} = \frac{-k_d + \sqrt{k_d^2 - 4mk_p}}{2m} = \frac{-12 + \sqrt{144-100}}{2} = \frac{-12 + 6.63}{2} \approx -2.68$$

**Time constant:** $\tau = \frac{1}{|\lambda|} \approx 0.37$ seconds

**99% settling time:** $t_{99} \approx 5\tau \approx 1.85$ seconds

At 30 fps, that's about **55 frames** to reach 99% of the way.

With `TRANSITION_FRAMES = 150` (5 seconds), drones have **plenty of time** to settle.

---

# 7. VIDEO OUTPUT FOR SUBTASK 1

## 7.1 Frame Count

| Phase | Frames | Duration at 30fps |
|-------|--------|-------------------|
| Transition | 150 | 5.0 seconds |
| Hold | 40 | 1.3 seconds |
| **Total** | **190** | **6.3 seconds** |

## 7.2 What You See

1. **Frames 0-30:** Drones accelerate from rest, start moving toward targets
2. **Frames 30-90:** Drones reach maximum speed, flying toward name shape
3. **Frames 90-120:** Drones slow down as they approach targets
4. **Frames 120-150:** Fine adjustment, settling into final positions
5. **Frames 150-190:** Hold - name shape clearly visible, minimal movement

## 7.3 Output Files

```
output/drone_show_math_phase1.mp4  - Only Subtask 1
output/drone_show_math_combined.mp4 - Full video (includes this)
```

---

# 8. SUMMARY

## Subtask 1 Pipeline:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Grid Positions │ ──→ │ Extract Targets  │ ──→ │ Greedy Assign   │
│  (Initial)      │     │ (Edge Detection) │     │ (Drone→Target)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ↓
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  RK4 Simulation │ ──→ │  Record Frames   │ ──→ │  Render Video   │
│  (150 steps)    │     │  (positions)     │     │  (MP4 output)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Key Parameters for Subtask 1:

| Parameter | Value | Effect |
|-----------|-------|--------|
| NUM_DRONES | 1200 | Detail level of name shape |
| TRANSITION_FRAMES | 150 | Time to reach target (5s) |
| HOLD_FRAMES | 40 | Pause duration (1.3s) |
| K_P | 25 | How fast drones "snap" to target |
| K_D | 12 | How smooth the motion is |
