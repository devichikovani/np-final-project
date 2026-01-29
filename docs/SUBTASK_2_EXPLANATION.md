# Subtask 2: Handwritten Name → Greeting

## Mathematical Analysis of Static-to-Static Transition

---

# 1. OVERVIEW

**Goal:** Transition drones from the handwritten name formation to display a greeting message.

**IVP Model (Static Target):**
```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T_new - x) + Frep - Kd*v]

Initial conditions:
x(0) = name_formation_position
v(0) = current_velocity  (may be near-zero after holding)
```

---

# 2. INITIAL STATE: HANDWRITTEN NAME

## 2.1 Starting Positions

After Subtask 1 + hold period:
```
x(0) ≈ T_name  (settled at name formation)
v(0) ≈ 0       (velocities damped out)
```

## 2.2 Why v(0) ≈ 0?

The hold period (HOLD_FRAMES = 80) gives drones time to settle:
```
Hold duration = 80 frames / 60 fps ≈ 1.33 seconds
```

With settling time τ ≈ 0.37s, the system reaches equilibrium.

---

# 3. TARGET EXTRACTION: GREETING MESSAGE

## 3.1 Text Rendering

```python
def render_text_to_image(text, width, height):
    """Render greeting text to binary image."""
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Calculate font size to fit
    font_size = estimate_font_size(text, width, height)
    font = ImageFont.truetype(FONT_PATH, font_size)
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    y = (height - (bbox[3] - bbox[1])) // 2
    
    draw.text((x, y), text, fill=255, font=font)
    return np.array(img)
```

## 3.2 Edge Detection

Same pipeline as Subtask 1:
```
Text Image → Grayscale → Canny → Point Sampling
```

---

# 4. REASSIGNMENT PROBLEM

## 4.1 The Challenge

Drones are already at name formation. Need to reassign to greeting targets.

**Constraint:** Minimize total travel distance.

## 4.2 Distance Matrix

```
D[i,j] = ||current_position[i] - greeting_target[j]||
```

## 4.3 Greedy Solution

```python
def reassign_targets(current_positions, new_targets):
    """Reassign drones to new targets minimizing distance."""
    dist = euclidean_distance_matrix(current_positions, new_targets)
    
    assignment = np.zeros((len(current_positions), 2))
    assigned = np.zeros(len(new_targets), dtype=bool)
    
    for i in range(len(current_positions)):
        masked_dist = dist[i].copy()
        masked_dist[assigned] = np.inf
        j = np.argmin(masked_dist)
        assignment[i] = new_targets[j]
        assigned[j] = True
    
    return assignment
```

---

# 5. THE IVP MODEL (IDENTICAL TO SUBTASK 1)

## 5.1 State Equations

```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T_greeting - x) + Frep - Kd*v]
```

## 5.2 Force Components

| Force | Formula | Purpose |
|-------|---------|---------|
| Spring | Kp(T-x) | Pull toward greeting target |
| Repulsion | Σ Krep(xi-xj)/d³ | Collision avoidance |
| Damping | -Kd*v | Energy dissipation |

---

# 6. TRANSITION DYNAMICS

## 6.1 Phase Portrait

The system evolves through:

1. **Acceleration Phase:** Drones accelerate toward new targets
2. **Cruise Phase:** Drones at Vmax (if distance is large)
3. **Deceleration Phase:** Drones slow down approaching targets
4. **Settling Phase:** Final convergence

## 6.2 Velocity Profile

For a drone traveling distance d:
```
Peak velocity: min(Vmax, √(2*Kp*d/m))
```

With Kp = 25, m = 1, Vmax = 100:
```
Saturation occurs when d > Vmax²/(2*Kp) = 10000/50 = 200 pixels
```

---

# 7. COLLISION HANDLING

## 7.1 Spatial Hashing

Grid-based acceleration for O(n) complexity:
```python
cell_size = 2 * R_SAFE  # 8 pixels
grid = spatial_hash_grid(positions, cell_size)
```

## 7.2 Repulsion Calculation

```python
def repulsion(self):
    """Compute collision avoidance forces."""
    forces = np.zeros_like(self.pos)
    grid, cell_idx = spatial_hash_grid(self.pos, 2*R_SAFE)
    
    for i in range(len(self.pos)):
        for neighbor in get_neighbors(grid, cell_idx[i]):
            if neighbor != i:
                diff = self.pos[i] - self.pos[neighbor]
                dist = np.linalg.norm(diff)
                
                if dist < R_SAFE and dist > 0:
                    forces[i] += K_REP * diff / (dist**3)
    
    return forces
```

---

# 8. PATH COMPLEXITY

## 8.1 Crossing Paths

When transitioning from name to greeting:
- Some drones move left
- Some drones move right
- Paths may cross

## 8.2 Repulsion Resolution

The repulsion force naturally resolves conflicts:
```
F_rep = Krep × (xi - xj) / |xi - xj|³
```

This scales as 1/d², creating strong separation at close range.

---

# 9. CONVERGENCE ANALYSIS

## 9.1 Lyapunov Function

Total energy of the system:
```
V = Σᵢ [½m|vᵢ|² + ½Kp|Tᵢ-xᵢ|²]
```

**Time derivative:**
```
dV/dt = Σᵢ [m*vᵢ·v̇ᵢ + Kp(Tᵢ-xᵢ)·(-vᵢ)]
      = Σᵢ [vᵢ·(Kp(Tᵢ-xᵢ) + Frep - Kd*vᵢ) - Kp*vᵢ·(Tᵢ-xᵢ)]
      = Σᵢ [vᵢ·Frep - Kd|vᵢ|²]
```

For well-separated drones (Frep ≈ 0):
```
dV/dt = -Kd Σᵢ|vᵢ|² ≤ 0
```

**Energy decreases monotonically → System converges.**

---

# 10. FRAME TIMING

| Parameter | Value | Meaning |
|-----------|-------|---------|
| TRANSITION_FRAMES | 300 | Frames for name→greeting |
| HOLD_FRAMES | 80 | Frames to hold greeting |
| STEPS_PER_FRAME | 10 | Physics steps per frame |
| DT | 0.05 | Time step |

**Simulation time per transition:**
```
300 frames × 10 steps × 0.05s = 150 seconds (sim time)
```

**Video duration:**
```
300 frames / 60 fps = 5 seconds
```

---

# 11. COMPARISON: SUBTASK 1 vs SUBTASK 2

| Aspect | Subtask 1 | Subtask 2 |
|--------|-----------|-----------|
| Initial State | Grid (uniform) | Name formation |
| Initial Velocity | 0 | ≈ 0 (after hold) |
| Target | Handwritten name | Greeting text |
| IVP Model | Same | Same |
| Path Complexity | Low (parallel) | Medium (crossing) |

---

# 12. SUMMARY

**Subtask 2 solves the same static-target IVP as Subtask 1:**

```
dx/dt = v * min(1, Vmax/|v|)
dv/dt = (1/m)[Kp(T - x) + Frep - Kd*v]
```

**Key differences from Subtask 1:**
- Non-zero initial state (coming from name formation)
- More complex path crossings (resolved by repulsion)
- Same convergence properties

**No velocity matching needed** — targets are static.
