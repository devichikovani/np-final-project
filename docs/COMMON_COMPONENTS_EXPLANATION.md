# Common Components: Shared Mathematical Foundations

## Physics Engine, Numerical Methods, and Image Processing

---

# 1. OVERVIEW

This document covers all code components that are **shared across all three subtasks**:

| Component | Used In | Description |
|-----------|---------|-------------|
| Physics Constants | All subtasks | M, Kp, Kd, Kv, Krep, Vmax, etc. |
| IVP Model | All subtasks | Spring-damper with velocity matching |
| RK4 Integration | All subtasks | 4th-order Runge-Kutta solver |
| Repulsion Forces | All subtasks | Collision avoidance |
| Spatial Hashing | All subtasks | O(n) neighbor detection |
| Distance Matrix | All subtasks | Euclidean distance computation |
| Greedy Assignment | All subtasks | Drone-to-target pairing |
| Edge Detection (Canny) | All subtasks | Target point extraction |
| Gaussian Blur | Image processing | Noise reduction |
| Sobel Gradients | Image processing | Edge detection |
| Otsu Thresholding | Image processing | Binary segmentation |
| Video Rendering | Output generation | MP4 creation |

---

# 2. PHYSICS CONSTANTS

```python
M = 1.0           # Mass
V_MAX = 100.0     # Max velocity (pixels/second)
K_P = 25.0        # Position gain (spring constant)
K_D = 12.0        # Damping coefficient
K_V = 8.0         # Velocity matching gain (for dynamic tracking)
K_REP = 50.0      # Repulsion strength
R_SAFE = 4.0      # Safety radius for collision avoidance
DT = 0.05         # Time step for integration
W, H = 800, 600   # Canvas size
```

## 2.1 Physical Meaning

### Mass (M = 1.0)

The inertia of each drone. From Newton's Second Law:
```
F = ma  →  a = F/m
```

| Value | Effect |
|-------|--------|
| M = 0.5 | Faster acceleration, more responsive |
| M = 1.0 | Balanced (default) |
| M = 2.0 | Slower acceleration, more sluggish |

### Maximum Velocity (V_MAX = 100.0)

Speed limit in pixels per second. Prevents unrealistic motion.

**Velocity saturation formula:**
```
v_effective = v * min(1, Vmax/|v|)
```

### Spring Constant (K_P = 25.0)

Stiffness of the virtual spring connecting each drone to its target.

**Spring force:**
```
F_spring = Kp * (T - x)
```

| Value | Effect |
|-------|--------|
| Kp = 10 | Weak pull, slow approach |
| Kp = 25 | Balanced (default) |
| Kp = 50 | Strong pull, fast snap |

### Damping Coefficient (K_D = 12.0)

Friction/drag that opposes motion.

**Damping force:**
```
F_damp = -Kd * v
```

**Damping Ratio Analysis:**
```
ζ = Kd / (2*sqrt(m * Kp)) = 12 / (2*sqrt(1 × 25)) = 1.2
```

Since ζ > 1: **Overdamped** — no oscillation, smooth approach.

### Velocity Matching Gain (K_V = 8.0) - NEW

**Used in Subtask 3 for dynamic tracking.**

Makes drone velocity match target velocity for predictive tracking.

**Velocity matching force:**
```
F_velocity = Kv * (dT/dt - v)
```

| Value | Effect |
|-------|--------|
| Kv = 0 | No velocity matching (laggy tracking) |
| Kv = 8 | Balanced (default) |
| Kv = 16 | Strong matching (may overshoot) |

### Repulsion Strength (K_REP = 50.0)

Force that prevents drone collisions.

**Repulsion force:**
```
F_rep = Krep * (xi - xj) / |xi - xj|³   if |xi - xj| < R_safe
```

---

# 3. THE COMPLETE IVP MODEL

## 3.1 Static Targets (Subtasks 1 & 2)

```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T - x) + Frep - Kd*v]
```

## 3.2 Dynamic Targets (Subtask 3)

```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T(t) - x) + Kv(dT/dt - v) + Frep - Kd*v]
```

The Kv term provides **velocity matching** for smooth dynamic tracking.

## 3.3 Force Breakdown

| Force | Formula | Purpose |
|-------|---------|---------|
| Spring | Kp(T-x) | Pull toward target |
| Velocity Match | Kv(dT/dt - v) | Match target velocity |
| Repulsion | Σ Krep(xi-xj)/d³ | Avoid collisions |
| Damping | -Kd*v | Dissipate energy |

---

# 4. RK4 INTEGRATION

## 4.1 The Algorithm

4th-order Runge-Kutta provides O(h⁴) accuracy:

```
k1 = f(t, y)
k2 = f(t + h/2, y + h*k1/2)
k3 = f(t + h/2, y + h*k2/2)
k4 = f(t + h, y + h*k3)
y_next = y + (h/6)(k1 + 2*k2 + 2*k3 + k4)
```

## 4.2 Implementation

```python
def step(self):
    """RK4 integration of IVP equations."""
    def deriv(p, v):
        """Compute dx/dt and dv/dt."""
        # Velocity saturation
        vnorm = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
        vnorm = np.maximum(vnorm, 1e-6)
        dx = v * np.minimum(1.0, V_MAX / vnorm)
        
        # Forces
        spring_force = K_P * (self.tgt - p)
        
        if self.use_velocity_matching:
            velocity_match_force = K_V * (self.tgt_vel - v)
        else:
            velocity_match_force = 0
        
        rep = self.repulsion()
        damping = K_D * v
        
        dv = (spring_force + velocity_match_force + rep - damping) / M
        return dx, dv
    
    # RK4 stages
    k1x, k1v = deriv(self.pos, self.vel)
    k2x, k2v = deriv(self.pos + 0.5*DT*k1x, self.vel + 0.5*DT*k1v)
    k3x, k3v = deriv(self.pos + 0.5*DT*k2x, self.vel + 0.5*DT*k2v)
    k4x, k4v = deriv(self.pos + DT*k3x, self.vel + DT*k3v)
    
    # Update state
    self.pos += (DT/6) * (k1x + 2*k2x + 2*k3x + k4x)
    self.vel += (DT/6) * (k1v + 2*k2v + 2*k3v + k4v)
```

## 4.3 Why RK4?

| Method | Order | Error | Stability |
|--------|-------|-------|-----------|
| Euler | 1 | O(h) | Poor |
| RK2 | 2 | O(h²) | Fair |
| RK4 | 4 | O(h⁴) | Excellent |

---

# 5. SPATIAL HASHING FOR COLLISION DETECTION

## 5.1 Problem

Brute-force collision detection: O(n²) — too slow for 2000 drones.

## 5.2 Solution: Spatial Hashing

Discretize space into cells. Only check neighbors in adjacent cells.

```python
def spatial_hash_grid(positions, cell_size):
    """Build spatial hash grid for O(n) neighbor queries."""
    cells = {}
    cell_indices = (positions / cell_size).astype(int)
    
    for i, idx in enumerate(cell_indices):
        key = tuple(idx)
        if key not in cells:
            cells[key] = []
        cells[key].append(i)
    
    return cells, cell_indices
```

## 5.3 Complexity

| Approach | Time Complexity |
|----------|-----------------|
| Brute Force | O(n²) |
| Spatial Hashing | O(n) average |

---

# 6. GREEDY ASSIGNMENT

## 6.1 Problem

Match n drones to n targets to minimize total distance.

## 6.2 Algorithm

```python
def greedy_assignment(positions, targets):
    """For each drone, find nearest unassigned target."""
    n = len(positions)
    dist = euclidean_distance_matrix(positions, targets)
    
    result = np.zeros((n, 2))
    assigned = np.zeros(n, dtype=bool)
    
    for i in range(n):
        masked_dist = dist[i].copy()
        masked_dist[assigned] = np.inf
        j = np.argmin(masked_dist)
        result[i] = targets[j]
        assigned[j] = True
    
    return result
```

## 6.3 Complexity

- Distance Matrix: O(n²)
- Assignment Loop: O(n²)
- Total: O(n²)

---

# 7. EUCLIDEAN DISTANCE MATRIX

## 7.1 Formula

```
d(a,b) = sqrt(Σ(a_i - b_i)²)
```

## 7.2 Optimized Computation

Using the identity: ||a-b||² = ||a||² + ||b||² - 2*a·b

```python
def euclidean_distance_matrix(A, B):
    A_sq = np.sum(A**2, axis=1, keepdims=True)
    B_sq = np.sum(B**2, axis=1, keepdims=True)
    AB = A @ B.T
    dist_sq = A_sq + B_sq.T - 2 * AB
    dist_sq = np.maximum(dist_sq, 0)  # Numerical stability
    return np.sqrt(dist_sq)
```

---

# 8. IMAGE PROCESSING

## 8.1 Canny Edge Detection

**Steps:**
1. Gaussian blur (noise reduction)
2. Sobel gradients (edge detection)
3. Non-maximum suppression (edge thinning)
4. Hysteresis thresholding (edge connection)

```python
def canny_edge_detection(image, low_thresh=50, high_thresh=150):
    blurred = gaussian_blur(image, kernel_size=5, sigma=1.4)
    magnitude, direction = sobel_gradients(blurred)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    thin_edges = non_maximum_suppression(magnitude, direction)
    edges = hysteresis_threshold(thin_edges, low_thresh, high_thresh)
    return edges
```

## 8.2 Sobel Operators

**Kernels:**
```
Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
      [-2, 0, 2],            [ 0,  0,  0],
      [-1, 0, 1]]            [ 1,  2,  1]]
```

**Gradient:**
```
magnitude = sqrt(Gx² + Gy²)
direction = atan2(Gy, Gx)
```

## 8.3 Gaussian Blur

**2D Gaussian kernel:**
```
G(x,y) = (1/(2π*σ²)) * exp(-(x² + y²)/(2*σ²))
```

## 8.4 Otsu's Thresholding

Automatically find optimal threshold by maximizing between-class variance:

```
σ_b²(t) = w0(t) * w1(t) * (μ0(t) - μ1(t))²
```

---

# 9. VIDEO RENDERING

```python
def render_video(frames, path, fps=60, label_override=None):
    """Render frames to MP4."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    for i, pos in enumerate(frames):
        frame = np.zeros((H, W, 3), np.uint8)
        
        for p in pos:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(frame, (x, y), 3, (40, 80, 80), -1)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # Add label
        cv2.putText(frame, label, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
```

---

# 10. SUMMARY: IVP EQUATIONS

## Static Targets (Subtasks 1 & 2):
```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T - x) + Frep - Kd*v]
```

## Dynamic Targets with Velocity Matching (Subtask 3):
```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T(t) - x) + Kv(dT/dt - v) + Frep - Kd*v]
```

The **velocity matching term Kv(dT/dt - v)** is the key innovation for smooth dynamic tracking.
