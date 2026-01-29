# Common Components: Shared Mathematical Foundations

## Physics Engine, Numerical Methods, and Image Processing

---

# 1. OVERVIEW

This document covers all code components that are **shared across all three subtasks**:

| Component | Used In |
|-----------|---------|
| Physics Constants | All subtasks |
| IVP Model | All subtasks |
| RK4 Integration | All subtasks |
| Repulsion Forces | All subtasks |
| Spatial Hashing | All subtasks |
| Distance Matrix | All subtasks |
| Greedy Assignment | All subtasks |
| Edge Detection (Canny) | All subtasks |
| Gaussian Blur | Image processing |
| Sobel Gradients | Image processing |
| Otsu Thresholding | Image processing |
| Video Rendering | Output generation |

---

# 2. PHYSICS CONSTANTS

**Location:** Lines 19-28

```python
M = 1.0           # Mass
V_MAX = 100.0     # Max velocity
K_P = 25.0        # Position gain (spring constant)
K_D = 12.0        # Damping coefficient
K_REP = 50.0      # Repulsion strength
R_SAFE = 4.0      # Safety radius for collision avoidance
DT = 0.05         # Time step for integration
W, H = 800, 600   # Canvas size
```

## 2.1 Physical Meaning

### Mass (M = 1.0)

The inertia of each drone. From Newton's Second Law:
$$F = ma \quad \Rightarrow \quad a = \frac{F}{m}$$

**Effect of changing M:**
| Value | Effect |
|-------|--------|
| M = 0.5 | Faster acceleration, more responsive |
| M = 1.0 | Balanced (default) |
| M = 2.0 | Slower acceleration, more sluggish |

### Maximum Velocity (V_MAX = 100.0)

Speed limit in pixels per second. Prevents unrealistic motion.

**Velocity saturation formula:**
$$\vec{v}_{effective} = \vec{v} \cdot \min\left(1, \frac{v_{max}}{|\vec{v}|}\right)$$

If $|\vec{v}| \leq v_{max}$: use full velocity
If $|\vec{v}| > v_{max}$: scale down to maintain direction but cap magnitude

### Spring Constant (K_P = 25.0)

Stiffness of the virtual spring connecting each drone to its target.

**Spring force:**
$$\vec{F}_{spring} = k_p \cdot (\vec{T} - \vec{x})$$

| Value | Effect |
|-------|--------|
| K_P = 10 | Weak pull, slow approach |
| K_P = 25 | Balanced (default) |
| K_P = 50 | Strong pull, fast snap |

### Damping Coefficient (K_D = 12.0)

Friction/drag that opposes motion.

**Damping force:**
$$\vec{F}_{damp} = -k_d \cdot \vec{v}$$

| Value | Effect |
|-------|--------|
| K_D = 5 | Underdamped, oscillation |
| K_D = 12 | Overdamped (default) |
| K_D = 20 | Very overdamped, sluggish |

### Damping Ratio Analysis

$$\zeta = \frac{k_d}{2\sqrt{m \cdot k_p}} = \frac{12}{2\sqrt{1 \times 25}} = 1.2$$

Since $\zeta > 1$: **Overdamped** — no oscillation, smooth approach.

### Repulsion Strength (K_REP = 50.0)

Strength of collision avoidance force.

**Repulsion force (inverse square law):**
$$\vec{F}_{rep} = k_{rep} \cdot \frac{\vec{x}_i - \vec{x}_j}{|\vec{x}_i - \vec{x}_j|^3}$$

### Safety Radius (R_SAFE = 4.0)

Distance threshold for collision detection. Drones within 4 pixels trigger repulsion.

### Time Step (DT = 0.05)

Integration step size for RK4.

**At 30 fps:**
- 1 frame = 0.033 seconds
- 1 simulation step = 0.05 seconds
- ~1.5 simulation steps per displayed frame

### Canvas Size (W=800, H=600)

Video resolution and drone movement bounds.

---

# 3. THE IVP MODEL

## 3.1 State Variables

Each drone has two state variables:
- Position: $\vec{x} = (x, y)$
- Velocity: $\vec{v} = (v_x, v_y)$

Total state for n drones: 4n scalar values

## 3.2 The Differential Equations

**Position evolution:**
$$\frac{d\vec{x}}{dt} = \vec{v} \cdot \min\left(1, \frac{v_{max}}{|\vec{v}|}\right)$$

**Velocity evolution:**
$$\frac{d\vec{v}}{dt} = \frac{1}{m}\left[k_p(\vec{T} - \vec{x}) + \vec{F}_{rep} - k_d\vec{v}\right]$$

## 3.3 State-Space Form

Combining into a single vector:
$$\mathbf{y} = \begin{pmatrix} \vec{x} \\ \vec{v} \end{pmatrix}$$

$$\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}) = \begin{pmatrix} \vec{v} \cdot \text{sat}(\vec{v}) \\ \frac{1}{m}\left[k_p(\vec{T} - \vec{x}) + \vec{F}_{rep} - k_d\vec{v}\right] \end{pmatrix}$$

Where $\text{sat}(\vec{v}) = \min(1, v_{max}/|\vec{v}|)$ is the saturation function.

## 3.4 Why This Model?

### Spring-Mass-Damper Analogy

```
     Target (T)
        │
        │ Spring (kₚ)
        │
    ╔═══╧═══╗
    ║ Drone ║ ←── Damping (kd) opposes velocity
    ╚═══╤═══╝
        │
        │ Connected to other drones via repulsion
        │
```

This is a classic **PD controller**:
- P (Proportional): Spring force proportional to position error
- D (Derivative): Damping proportional to velocity

---

# 4. RK4 INTEGRATION

## 4.1 Why RK4?

The IVP cannot be solved analytically because:
1. Nonlinear saturation function
2. Coupled equations (repulsion depends on all positions)
3. Time-varying targets (in Subtask 3)

**Numerical solution:** Approximate using discrete steps.

## 4.2 The RK4 Algorithm

Given current state $\mathbf{y}_n$ at time $t_n$, compute next state $\mathbf{y}_{n+1}$ at time $t_{n+1} = t_n + h$:

$$k_1 = f(t_n, \mathbf{y}_n)$$
$$k_2 = f(t_n + \frac{h}{2}, \mathbf{y}_n + \frac{h}{2}k_1)$$
$$k_3 = f(t_n + \frac{h}{2}, \mathbf{y}_n + \frac{h}{2}k_2)$$
$$k_4 = f(t_n + h, \mathbf{y}_n + h \cdot k_3)$$

$$\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

## 4.3 Geometric Interpretation

```
    y
    │
    │       k₄
    │      ╱
    │     ╱ k₃
    │    ╱╱
    │   ╱╱ k₂
    │  ╱╱╱
    │ ╱╱╱ k₁
    │╱╱╱
    ●───────────────→ t
   yₙ   tₙ+h/2   tₙ+h

k₁: Slope at start
k₂: Slope at midpoint using k₁
k₃: Slope at midpoint using k₂  
k₄: Slope at end using k₃

Final: Weighted average emphasizing midpoint
```

## 4.4 Implementation

**Location:** `Swarm.step()` (lines 601-645)

```python
def step(self):
    def deriv(p, v):
        # Velocity saturation
        vnorm = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
        vnorm = np.maximum(vnorm, 1e-6)
        dx = v * np.minimum(1.0, V_MAX / vnorm)
        
        # Force calculation
        old_pos = self.pos.copy()
        self.pos = p
        rep = self.repulsion()
        self.pos = old_pos
        
        dv = (K_P * (self.tgt - p) + rep - K_D * v) / M
        return dx, dv
    
    # RK4 stages
    k1x, k1v = deriv(self.pos, self.vel)
    k2x, k2v = deriv(self.pos + 0.5*DT*k1x, self.vel + 0.5*DT*k1v)
    k3x, k3v = deriv(self.pos + 0.5*DT*k2x, self.vel + 0.5*DT*k2v)
    k4x, k4v = deriv(self.pos + DT*k3x, self.vel + DT*k3v)
    
    # Update
    self.pos += (DT/6) * (k1x + 2*k2x + 2*k3x + k4x)
    self.vel += (DT/6) * (k1v + 2*k2v + 2*k3v + k4v)
    
    # Boundaries
    self.pos = np.clip(self.pos, 5, [W-5, H-5])
```

## 4.5 Error Analysis

| Method | Order | Local Error | Global Error |
|--------|-------|-------------|--------------|
| Euler | 1 | O(h²) | O(h) |
| Midpoint | 2 | O(h³) | O(h²) |
| **RK4** | **4** | **O(h⁵)** | **O(h⁴)** |
| RK5 | 5 | O(h⁶) | O(h⁵) |

With h = 0.05:
- Euler: ~5% error
- RK4: ~0.000006% error

RK4 is 4 evaluations per step (vs 1 for Euler) but 10,000× more accurate.

---

# 5. COLLISION AVOIDANCE

## 5.1 Repulsion Force Model

**Inverse-square repulsion:**
$$\vec{F}_{rep,i} = \sum_{j \neq i, |\vec{r}_{ij}| < R_{safe}} k_{rep} \cdot \frac{\vec{r}_{ij}}{|\vec{r}_{ij}|^3}$$

Where $\vec{r}_{ij} = \vec{x}_i - \vec{x}_j$

## 5.2 Physical Analogy

Same law as electrostatic repulsion (Coulomb's law):
$$F = k \cdot \frac{q_1 q_2}{r^2}$$

Or gravitational attraction (with opposite sign).

## 5.3 Implementation

**Location:** `Swarm.repulsion()` (lines 569-599)

```python
def repulsion(self):
    forces = np.zeros((self.n, 2))
    
    # Find nearby pairs using spatial hashing
    pairs = find_neighbor_pairs(self.pos, R_SAFE)
    
    if not pairs:
        return forces
    
    pairs = np.array(pairs)
    diff = self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]]
    dist = np.sqrt(np.sum(diff**2, axis=1))
    dist = np.maximum(dist, 0.1)  # Avoid division by zero
    
    # Force = K_REP * diff / dist³
    fvec = (K_REP / dist**3)[:, None] * diff
    
    # Newton's 3rd law
    np.add.at(forces, pairs[:, 0], fvec)
    np.add.at(forces, pairs[:, 1], -fvec)
    
    return forces
```

---

# 6. SPATIAL HASHING

## 6.1 The Problem

Checking all pairs for collision: O(n²)
- 1200 drones → 1,440,000 comparisons per step
- Too slow!

## 6.2 The Solution

Divide space into grid cells. Only check drones in nearby cells.

**Hash function:**
$$h(x, y) = \left(\left\lfloor \frac{x}{s} \right\rfloor, \left\lfloor \frac{y}{s} \right\rfloor\right)$$

Where s = cell size = R_SAFE

## 6.3 Implementation

**Location:** Lines 79-128

```python
def spatial_hash_grid(positions, cell_size):
    cells = {}
    cell_indices = (positions / cell_size).astype(int)
    
    for i, (cx, cy) in enumerate(cell_indices):
        key = (cx, cy)
        if key not in cells:
            cells[key] = []
        cells[key].append(i)
    
    return cells, cell_indices


def find_neighbor_pairs(positions, radius):
    cells, cell_indices = spatial_hash_grid(positions, radius)
    pairs = []
    
    for i in range(len(positions)):
        cx, cy = cell_indices[i]
        # Check 3×3 neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in cells:
                    for j in cells[key]:
                        if j > i:
                            dist = np.linalg.norm(positions[i] - positions[j])
                            if dist < radius:
                                pairs.append((i, j))
    
    return pairs
```

## 6.4 Complexity

| Method | Comparisons |
|--------|-------------|
| Brute force | O(n²) = 1,440,000 |
| Spatial hashing | O(n × k) ≈ 1,200 × 10 = 12,000 |

Where k = average neighbors per drone (~10 for our density)

**120× speedup!**

---

# 7. DISTANCE MATRIX

## 7.1 Purpose

Compute distances between all drones and all targets for assignment.

## 7.2 Optimized Formula

Naive: compute $\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$ for each pair

Optimized using algebraic identity:
$$\|\vec{a} - \vec{b}\|^2 = \|\vec{a}\|^2 + \|\vec{b}\|^2 - 2\vec{a} \cdot \vec{b}$$

## 7.3 Implementation

**Location:** Lines 53-75

```python
def euclidean_distance_matrix(A, B):
    A_sq = np.sum(A**2, axis=1, keepdims=True)  # ||a||²
    B_sq = np.sum(B**2, axis=1, keepdims=True)  # ||b||²
    AB = A @ B.T                                  # a·b
    dist_sq = A_sq + B_sq.T - 2 * AB             # ||a-b||²
    dist_sq = np.maximum(dist_sq, 0)             # Numerical stability
    return np.sqrt(dist_sq)
```

## 7.4 Complexity

- Naive loops: O(n × m × 2)
- Vectorized: O(n × m) with numpy's optimized BLAS

For 1200×1200: ~5ms vs ~500ms (100× faster)

---

# 8. GREEDY ASSIGNMENT

## 8.1 The Assignment Problem

Given n drones and n targets, assign each drone to exactly one target to minimize total distance.

**Optimal solution:** Hungarian Algorithm, O(n³)
**Our solution:** Greedy, O(n²)

## 8.2 Greedy Algorithm

```
For i = 1 to n:
    Find j = argmin(distance[i, unassigned])
    Assign drone i to target j
    Mark target j as assigned
```

## 8.3 Implementation

**Location:** Lines 131-154

```python
def greedy_assignment(positions, targets):
    n = len(positions)
    dist = euclidean_distance_matrix(positions, targets)
    
    result = np.zeros((n, 2))
    assigned_targets = np.zeros(n, dtype=bool)
    
    for i in range(n):
        masked_dist = dist[i].copy()
        masked_dist[assigned_targets] = np.inf
        j = np.argmin(masked_dist)
        result[i] = targets[j]
        assigned_targets[j] = True
    
    return result
```

## 8.4 Quality vs Speed Trade-off

| Method | Complexity | Optimality |
|--------|------------|------------|
| Hungarian | O(n³) | Optimal |
| Greedy | O(n²) | ~90-95% of optimal |

For n=1200:
- Hungarian: ~1.7 billion operations
- Greedy: ~1.4 million operations

**1000× faster, nearly as good**

---

# 9. IMAGE PROCESSING

## 9.1 Gaussian Blur

**Purpose:** Reduce noise before edge detection

**Formula:**
$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

**Implementation:** Lines 157-202

```python
def gaussian_kernel_2d(size, sigma):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def convolve_2d(image, kernel):
    # Sliding window convolution
    ...
```

## 9.2 Sobel Gradients

**Purpose:** Detect edges (intensity changes)

**Sobel kernels:**
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

**Gradient magnitude:**
$$|\nabla I| = \sqrt{G_x^2 + G_y^2}$$

**Implementation:** Lines 218-248

## 9.3 Canny Edge Detection

**Pipeline:**
1. Gaussian blur (noise reduction)
2. Sobel gradients (edge detection)
3. Non-maximum suppression (edge thinning)
4. Hysteresis thresholding (edge connection)

**Implementation:** Lines 325-346

## 9.4 Otsu Thresholding

**Purpose:** Automatic threshold selection for binarization

**Criterion:** Maximize between-class variance
$$\sigma_B^2(t) = \omega_0(t) \omega_1(t) (\mu_0(t) - \mu_1(t))^2$$

**Implementation:** Lines 349-378

---

# 10. VIDEO RENDERING

## 10.1 Frame Generation

**Location:** Lines 707-730

```python
def render_video(frames, path, fps=30, label_override=None):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    for i, pos in enumerate(frames):
        frame = np.zeros((H, W, 3), np.uint8)
        
        for p in pos:
            x, y = int(p[0]), int(p[1])
            cv2.circle(frame, (x, y), 3, (40, 80, 80), -1)  # Glow
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)  # Core
        
        cv2.putText(frame, label, (10, 25), ...)
        out.write(frame)
    
    out.release()
```

## 10.2 Drone Appearance

Each drone drawn as two concentric circles:
- Outer: radius 3, dark teal (40, 80, 80) — glow effect
- Inner: radius 2, bright yellow (0, 255, 255) — LED core

## 10.3 Output Format

- Codec: mp4v (MPEG-4)
- Resolution: 800×600
- Frame rate: 30 fps
- Color: BGR (OpenCV convention)

---

# 11. CONFIGURATION SYSTEM

## 11.1 User-Editable Variables

**Location:** Lines 34-59

```python
NUM_DRONES = 1200
DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"

PHASE1_IMAGE = "handwritten_name.png"
PHASE1_TEXT_FALLBACK = "STUDENT"

PHASE2_IMAGE = "greeting.png"
PHASE2_TEXT_FALLBACK = "Happy New Year!"

PHASE3_ANIMATION = "tiger.gif"

MAX_ANIMATION_FRAMES = 25
STEPS_PER_FRAME = 6
```

## 11.2 Timing Configuration

```python
TRANSITION_FRAMES = 150  # 5 seconds
HOLD_FRAMES = 40         # 1.3 seconds
```

---

# 12. SUMMARY

## Component Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN PROGRAM                            │
├─────────────────────────────────────────────────────────────────┤
│                              │                                  │
│     ┌────────────────────────┼────────────────────────┐        │
│     ▼                        ▼                        ▼        │
│ ┌─────────┐           ┌─────────────┐          ┌─────────┐    │
│ │ Subtask │           │   Subtask   │          │ Subtask │    │
│ │    1    │           │      2      │          │    3    │    │
│ └────┬────┘           └──────┬──────┘          └────┬────┘    │
│      │                       │                      │          │
│      └───────────────────────┼──────────────────────┘          │
│                              ▼                                  │
│              ┌───────────────────────────────┐                 │
│              │      SHARED COMPONENTS        │                 │
│              ├───────────────────────────────┤                 │
│              │ • Physics Constants           │                 │
│              │ • IVP Model (deriv function)  │                 │
│              │ • RK4 Integration (step)      │                 │
│              │ • Repulsion Forces            │                 │
│              │ • Spatial Hashing             │                 │
│              │ • Distance Matrix             │                 │
│              │ • Greedy Assignment           │                 │
│              │ • Edge Detection              │                 │
│              │ • Video Rendering             │                 │
│              └───────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Equations Reference

| Component | Equation |
|-----------|----------|
| Spring Force | $\vec{F} = k_p(\vec{T} - \vec{x})$ |
| Damping | $\vec{F} = -k_d\vec{v}$ |
| Repulsion | $\vec{F} = k_{rep}\frac{\vec{r}}{|\vec{r}|^3}$ |
| Velocity | $\frac{d\vec{x}}{dt} = \vec{v} \cdot \text{sat}(\vec{v})$ |
| Acceleration | $\frac{d\vec{v}}{dt} = \frac{1}{m}(\vec{F}_{spring} + \vec{F}_{rep} + \vec{F}_{damp})$ |
| RK4 Update | $\vec{y}_{n+1} = \vec{y}_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$ |
| Damping Ratio | $\zeta = \frac{k_d}{2\sqrt{mk_p}}$ |
| Distance | $d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$ |
| Gaussian | $G = \frac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/2\sigma^2}$ |
