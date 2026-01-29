# Full Mathematical Explanation - Drone Show Simulation

## Complete Line-by-Line Analysis: Physics + Video Behavior

---

# 🔵 PART 1: PHYSICS CONSTANTS AND THEIR MEANING

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

### Physical Meaning:

| Parameter | Symbol | Physical Meaning | Video Effect |
|-----------|--------|------------------|--------------|
| `M = 1.0` | $m$ | Mass of each drone | Higher mass = slower acceleration, more sluggish movement |
| `V_MAX = 100.0` | $v_{max}$ | Maximum velocity cap | Prevents drones from moving too fast (blur/teleporting) |
| `K_P = 25.0` | $k_p$ | Spring stiffness | Higher = faster snap to target, lower = slower drift |
| `K_D = 12.0` | $k_d$ | Damping coefficient | Higher = less oscillation, more "brake" effect |
| `K_REP = 50.0` | $k_{rep}$ | Repulsion strength | Higher = drones push harder away from each other |
| `R_SAFE = 4.0` | $R_{safe}$ | Collision detection radius | Drones within this distance repel each other |
| `DT = 0.05` | $\Delta t$ | Integration time step | Smaller = more accurate but slower simulation |

---

# 🔵 PART 2: DISTANCE CALCULATIONS

## Lines 39-61: `euclidean_distance_matrix(A, B)`

### Purpose:
Calculate distances between every point in set A to every point in set B.

### Mathematical Formula:

**Basic Euclidean Distance:**
$$d(a, b) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

For 2D: $d = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2}$

**Optimized Formula (used in code):**
$$\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \cdot a \cdot b$$

### Line-by-Line:

```python
A_sq = np.sum(A**2, axis=1, keepdims=True)  # ||a||² for each point in A
```
- Computes $\|a\|^2 = x_a^2 + y_a^2$ for each drone position
- Result: Column vector of squared norms

```python
B_sq = np.sum(B**2, axis=1, keepdims=True)  # ||b||² for each point in B
```
- Same for target positions

```python
AB = A @ B.T  # Dot product matrix
```
- Matrix multiplication: $A \cdot B^T$ gives all pairwise dot products
- Entry $(i,j) = a_i \cdot b_j = x_{a_i} x_{b_j} + y_{a_i} y_{b_j}$

```python
dist_sq = A_sq + B_sq.T - 2 * AB  # ||a-b||² = ||a||² + ||b||² - 2*a·b
```
- Applies the algebraic identity
- Result: Matrix where entry $(i,j)$ is squared distance from drone $i$ to target $j$

```python
dist_sq = np.maximum(dist_sq, 0)  # Numerical stability
return np.sqrt(dist_sq)
```
- Clamps negative values (floating-point errors) to zero
- Takes square root to get actual distances

### Video Effect:
This matrix is used to find which target is closest to which drone. It determines the optimal assignment paths that you see as drones reorganize from one shape to another.

---

# 🔵 PART 3: SPATIAL HASHING (Collision Detection Optimization)

## Lines 64-82: `spatial_hash_grid(positions, cell_size)`

### Purpose:
Divide the 2D space into a grid of cells for fast neighbor lookup.

### Mathematical Concept:

**Hash Function:**
$$h(x, y) = \left(\lfloor x / s \rfloor, \lfloor y / s \rfloor\right)$$

Where $s$ = cell size (here, $R_{safe} = 4$)

### Line-by-Line:

```python
cells = {}
cell_indices = (positions / cell_size).astype(int)
```
- Divides each position by cell size and truncates to integer
- Example: position $(17.3, 25.8)$ with cell_size=4 → cell $(4, 6)$

```python
for i, (cx, cy) in enumerate(cell_indices):
    key = (cx, cy)
    if key not in cells:
        cells[key] = []
    cells[key].append(i)
```
- Groups drone indices by their cell
- Result: Dictionary mapping cell coordinates → list of drone indices

### Video Effect:
Without this, checking collisions is $O(n^2)$ - every drone vs every other drone. With spatial hashing, only nearby drones are checked, making it $O(n)$ average case. This allows smooth 30fps video with 1000+ drones.

---

## Lines 85-114: `find_neighbor_pairs(positions, radius)`

### Purpose:
Find all pairs of drones that are within collision distance.

### Algorithm:

1. Build spatial hash grid with cell_size = radius
2. For each drone, only check drones in same cell and 8 neighboring cells
3. Compute actual distance only for candidates

```python
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        key = (cx + dx, cy + dy)
```
- Checks 3×3 neighborhood of cells (9 cells total)
- This is because a drone at cell edge could be within radius of drone in adjacent cell

```python
diff = positions[i] - positions[j]
dist = np.sqrt(diff[0]**2 + diff[1]**2)
if dist < radius:
    pairs.append((i, j))
```
- Actual Euclidean distance check: $d = \sqrt{\Delta x^2 + \Delta y^2}$
- Only adds pair if truly within collision radius

### Video Effect:
This identifies which drones are too close and need to push away from each other. You see this as drones "avoiding" each other during transitions - they never overlap.

---

# 🔵 PART 4: ASSIGNMENT ALGORITHM

## Lines 117-140: `greedy_assignment(positions, targets)`

### Purpose:
Assign each drone to a target position with minimal total travel.

### Algorithm:
```
For i = 1 to n:
    Find j* = argmin_{j ∉ assigned} d(drone_i, target_j)
    Assign drone_i → target_j*
    Mark target_j* as assigned
```

### Line-by-Line:

```python
dist = euclidean_distance_matrix(positions, targets)
```
- Computes all pairwise distances (n×n matrix)

```python
for i in range(n):
    masked_dist = dist[i].copy()
    masked_dist[assigned_targets] = np.inf
```
- For drone $i$, mask out already-assigned targets with infinity
- This prevents multiple drones going to same target

```python
    j = np.argmin(masked_dist)  # Find nearest available target
    result[i] = targets[j]
    assigned_targets[j] = True
```
- Select closest available target
- Record assignment and mark target as taken

### Complexity:
- Greedy: $O(n^2)$ - fast but not globally optimal
- Hungarian Algorithm (optimal): $O(n^3)$ - slower but minimizes total distance

### Video Effect:
This determines the "pairing" of drones to target positions. Good assignment means short, non-crossing paths. Poor assignment causes drones to cross over each other, creating messy transitions.

**Visually:** When transitioning from "RAMAZ" to "Happy New Year", each letter's drones smoothly flow to their new positions rather than chaotically swapping.

---

# 🔵 PART 5: IMAGE PROCESSING MATHEMATICS

## Lines 143-157: `gaussian_kernel_2d(size, sigma)`

### Purpose:
Create a blur filter for noise reduction.

### Mathematical Formula:

**2D Gaussian Function:**
$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

### Line-by-Line:

```python
ax = np.arange(-size // 2 + 1, size // 2 + 1)
xx, yy = np.meshgrid(ax, ax)
```
- Creates coordinate grid centered at (0,0)
- For size=5: ax = [-1, 0, 1, 2, 3] (approximately centered)

```python
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
```
- Applies Gaussian formula at each grid point
- Center (0,0) has highest value, decreases radially

```python
return kernel / kernel.sum()
```
- Normalizes so weights sum to 1 (preserves image brightness)

### Example Kernel (size=3, σ=1):
```
[ 0.075  0.124  0.075 ]
[ 0.124  0.204  0.124 ]
[ 0.075  0.124  0.075 ]
```

### Video Effect:
Blurs the input image to reduce noise before edge detection. This prevents detecting noise as edges, giving cleaner drone formations.

---

## Lines 160-186: `convolve_2d(image, kernel)`

### Purpose:
Apply a filter to an image (sliding window operation).

### Mathematical Formula:

**2D Convolution:**
$$(f * g)(x, y) = \sum_{i} \sum_{j} f(i, j) \cdot g(x-i, y-j)$$

For each pixel, multiply overlapping regions and sum.

### Line-by-Line:

```python
padded = np.pad(image.astype(float), ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
```
- Adds border pixels (edge extension) to handle boundaries
- Without padding, edges would be cut off

```python
for i in range(kh):
    for j in range(kw):
        output += padded[i:i+h, j:j+w] * kernel[i, j]
```
- Shifts image by (i,j) and multiplies by kernel weight
- Accumulates all shifted-and-weighted versions

### Video Effect:
This is the core operation for all image filters - blur, edge detection, sharpening. It transforms the input image into something the algorithm can process.

---

## Lines 204-234: `sobel_gradients(image)`

### Purpose:
Detect edges by finding where pixel values change rapidly.

### Mathematical Concept:

**Gradient of Image:**
$$\nabla I = \left(\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right)$$

Edges occur where $|\nabla I|$ is large.

**Sobel Operators (approximates derivatives):**

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

### Line-by-Line:

```python
gx = convolve_2d(image, sobel_x)  # ∂I/∂x
gy = convolve_2d(image, sobel_y)  # ∂I/∂y
```
- Computes horizontal and vertical gradients

```python
magnitude = np.sqrt(gx**2 + gy**2)  # |∇I|
```
- Gradient magnitude: $|\nabla I| = \sqrt{G_x^2 + G_y^2}$
- High magnitude = strong edge

```python
direction = np.arctan2(gy, gx)  # Edge direction
```
- Angle of edge: $\theta = \arctan(G_y / G_x)$
- Used to determine edge orientation (horizontal, vertical, diagonal)

### Video Effect:
Extracts the outline of shapes from images. This is how drones know to form the outline of a tiger or text - they position themselves on detected edges.

---

## Lines 237-273: `non_maximum_suppression(magnitude, direction)`

### Purpose:
Thin edges to single-pixel width.

### Algorithm:

For each pixel:
1. Determine gradient direction (which way the edge "points")
2. Check the two neighbors in that direction
3. Keep pixel only if it's the local maximum

### Line-by-Line:

```python
angle = direction * 180 / np.pi
angle[angle < 0] += 180
```
- Convert radians to degrees (0-180°)
- Negative angles wrapped to positive

```python
angle_bin = np.zeros((h, w), dtype=int)
angle_bin[(angle >= 22.5) & (angle < 67.5)] = 1   # 45° diagonal
angle_bin[(angle >= 67.5) & (angle < 112.5)] = 2  # 90° vertical
angle_bin[(angle >= 112.5) & (angle < 157.5)] = 3 # 135° diagonal
```
- Quantize angles to 4 directions: 0°, 45°, 90°, 135°

```python
if m >= n1 and m >= n2:
    output[i-1, j-1] = m
```
- Keep pixel only if it's larger than both neighbors in gradient direction
- Otherwise suppress to 0

### Video Effect:
Without this, edges would be thick blobs. After NMS, edges are crisp 1-pixel lines - giving precise positions for drones.

---

## Lines 276-308: `hysteresis_threshold(image, low_thresh, high_thresh)`

### Purpose:
Connect edge fragments and remove noise.

### Algorithm:

**Double Thresholding:**
- Strong edges: pixels > high_thresh (definitely edges)
- Weak edges: low_thresh < pixels < high_thresh (maybe edges)
- Non-edges: pixels < low_thresh (definitely not edges)

**Edge Tracking:**
- Keep weak edges only if connected to strong edges

### Line-by-Line:

```python
strong_i, strong_j = np.where(image >= high_thresh)
weak_i, weak_j = np.where((image >= low_thresh) & (image < high_thresh))
```
- Classify pixels into strong/weak/none

```python
if output[i, j] == weak:
    neighbors = output[i-1:i+2, j-1:j+2]
    if np.any(neighbors == strong):
        output[i, j] = strong
    else:
        output[i, j] = 0
```
- 8-connected neighborhood check
- Weak edge becomes strong if touching any strong edge
- Otherwise removed

### Video Effect:
Produces clean, connected edges. The tiger outline is complete rather than fragmented, so drones form a recognizable shape.

---

## Lines 311-332: `canny_edge_detection(image, low_thresh, high_thresh)`

### Purpose:
Complete edge detection pipeline.

### Pipeline:
```
Input Image → Gaussian Blur → Sobel Gradients → Non-Maximum Suppression → Hysteresis
```

```python
blurred = gaussian_blur(image, kernel_size=5, sigma=1.4)
magnitude, direction = sobel_gradients(blurred)
thin_edges = non_maximum_suppression(magnitude.astype(float), direction)
edges = hysteresis_threshold(thin_edges, low_thresh, high_thresh)
```

### Video Effect:
This entire pipeline converts a photograph or GIF frame into drone target positions - the points where drones should fly to form the image.

---

## Lines 335-364: `otsu_threshold(image)`

### Purpose:
Automatically find the best threshold to separate foreground from background.

### Mathematical Principle:

**Maximize Between-Class Variance:**
$$\sigma_B^2(t) = \omega_0(t) \cdot \omega_1(t) \cdot (\mu_0(t) - \mu_1(t))^2$$

Where:
- $\omega_0, \omega_1$ = probability of each class (background/foreground)
- $\mu_0, \mu_1$ = mean intensity of each class
- $t$ = threshold value

### Line-by-Line:

```python
hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
hist = hist.astype(float) / hist.sum()
```
- Build histogram: count pixels at each intensity 0-255
- Normalize to probabilities

```python
for t in range(1, 256):
    w0 = hist[:t].sum()    # P(pixel ≤ t)
    w1 = hist[t:].sum()    # P(pixel > t)
```
- Try every possible threshold
- Calculate class probabilities

```python
    mu0 = np.sum(np.arange(t) * hist[:t]) / w0    # E[I | I ≤ t]
    mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1  # E[I | I > t]
```
- Calculate class means (weighted average intensity)

```python
    variance = w0 * w1 * (mu0 - mu1)**2
```
- Between-class variance: how "separated" the two classes are

```python
binary = np.where(image > best_thresh, 0, 255).astype(np.uint8)
```
- Apply optimal threshold to create binary image

### Video Effect:
When edge detection gives sparse points, Otsu thresholding provides a fallback - filling in the shape's interior with drone positions.

---

# 🔵 PART 6: PHYSICS SIMULATION (The Core IVP)

## Lines 489-503: `Swarm.__init__(self, n)`

### Purpose:
Initialize n drones on a grid pattern.

```python
cols = int(np.sqrt(n * W / H))
rows = int(np.ceil(n / cols))
xx, yy = np.meshgrid(np.linspace(80, W-80, cols), np.linspace(80, H-80, rows))
```
- Creates evenly spaced grid
- Aspect ratio matches canvas (more columns than rows for wide canvas)

### Video Effect:
At video start, you see drones arranged in a neat rectangular grid - this is the initial condition before any simulation.

---

## Lines 505-527: `Swarm.repulsion(self)`

### Purpose:
Compute collision avoidance forces between nearby drones.

### Mathematical Model:

**Inverse-Square Repulsive Force:**
$$\vec{f}_{rep} = k_{rep} \cdot \frac{\vec{x}_i - \vec{x}_j}{|\vec{x}_i - \vec{x}_j|^3}$$

This is similar to electrostatic/gravitational force but repulsive.

### Line-by-Line:

```python
pairs = find_neighbor_pairs(self.pos, R_SAFE)
```
- Find all drone pairs within safety radius (4 pixels)

```python
diff = self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]]  # x_i - x_j
dist = np.sqrt(np.sum(diff**2, axis=1))  # |x_i - x_j|
dist = np.maximum(dist, 0.1)  # Avoid division by zero
```
- Compute displacement vectors and distances

```python
fvec = (K_REP / dist**3)[:, None] * diff
```
- Force magnitude: $\frac{k_{rep}}{d^3}$
- Direction: unit vector $\frac{\vec{x}_i - \vec{x}_j}{|\vec{x}_i - \vec{x}_j|}$
- Combined: $\frac{k_{rep}}{d^3} \cdot (\vec{x}_i - \vec{x}_j) = \frac{k_{rep} \cdot (\vec{x}_i - \vec{x}_j)}{d^3}$

```python
np.add.at(forces, pairs[:, 0], fvec)   # Drone i feels force
np.add.at(forces, pairs[:, 1], -fvec)  # Drone j feels opposite force
```
- Newton's Third Law: Equal and opposite reactions

### Video Effect:
Drones push away from each other when too close. You see this as:
- No overlapping drones
- Smooth "flowing around" behavior during transitions
- Natural spacing in final formations

---

## Lines 529-573: `Swarm.step(self)` - THE CORE RK4 SOLVER

### Purpose:
Advance the simulation by one time step using 4th-order Runge-Kutta.

### The IVP (Initial Value Problem):

**State Variables:**
- $\vec{x}$ = position (x, y)
- $\vec{v}$ = velocity (vx, vy)

**Differential Equations:**

$$\frac{d\vec{x}}{dt} = \vec{v} \cdot \min\left(1, \frac{v_{max}}{|\vec{v}|}\right)$$

$$\frac{d\vec{v}}{dt} = \frac{1}{m}\left[k_p(\vec{T} - \vec{x}) + \vec{f}_{rep} - k_d\vec{v}\right]$$

### Physical Interpretation:

| Term | Formula | Meaning |
|------|---------|---------|
| Spring force | $k_p(\vec{T} - \vec{x})$ | Pulls drone toward target |
| Repulsion | $\vec{f}_{rep}$ | Pushes away from nearby drones |
| Damping | $-k_d\vec{v}$ | Slows drone down (friction) |
| Velocity saturation | $\min(1, v_{max}/|\vec{v}|)$ | Caps maximum speed |

### The `deriv()` Function:

```python
def deriv(p, v):
    # Velocity magnitude
    vnorm = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    vnorm = np.maximum(vnorm, 1e-6)
```
- Compute $|\vec{v}|$ for each drone
- Clamp to small positive value to avoid 0/0

```python
    # Velocity saturation: dx/dt = v * min(1, vmax/|v|)
    dx = v * np.minimum(1.0, V_MAX / vnorm)
```
- If $|\vec{v}| < v_{max}$: use full velocity
- If $|\vec{v}| > v_{max}$: scale down to cap at $v_{max}$
- This prevents drones from accelerating indefinitely

```python
    # Compute repulsion at position p
    old_pos = self.pos.copy()
    self.pos = p
    rep = self.repulsion()
    self.pos = old_pos
```
- Temporarily move drones to position p
- Compute what repulsion forces would be
- Restore original positions

```python
    # dv/dt = (1/m)[kp(T-x) + frep - kd*v]
    dv = (K_P * (self.tgt - p) + rep - K_D * v) / M
```
- Spring: $k_p(\vec{T} - \vec{x})$ points toward target
- Repulsion: prevents collisions
- Damping: $-k_d\vec{v}$ opposes motion
- Divide by mass to get acceleration

### RK4 Integration:

$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + h/2, y_n + h \cdot k_1/2)$$
$$k_3 = f(t_n + h/2, y_n + h \cdot k_2/2)$$
$$k_4 = f(t_n + h, y_n + h \cdot k_3)$$
$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

```python
k1x, k1v = deriv(self.pos, self.vel)
```
- Stage 1: Derivative at current state

```python
k2x, k2v = deriv(self.pos + 0.5*DT*k1x, self.vel + 0.5*DT*k1v)
```
- Stage 2: Derivative at half-step using k1

```python
k3x, k3v = deriv(self.pos + 0.5*DT*k2x, self.vel + 0.5*DT*k2v)
```
- Stage 3: Derivative at half-step using k2

```python
k4x, k4v = deriv(self.pos + DT*k3x, self.vel + DT*k3v)
```
- Stage 4: Derivative at full step using k3

```python
self.pos += (DT/6) * (k1x + 2*k2x + 2*k3x + k4x)
self.vel += (DT/6) * (k1v + 2*k2v + 2*k3v + k4v)
```
- Weighted average: middle stages count double
- This achieves 4th-order accuracy: error ~ $O(h^5)$ per step

```python
self.pos = np.clip(self.pos, 5, [W-5, H-5])
```
- Keep drones inside canvas boundaries

### Video Effect:
This is the "motion engine" of the entire simulation. Each frame:
1. Drones feel a spring force pulling them toward their target
2. Nearby drones push each other away
3. Damping prevents oscillation/overshooting
4. RK4 updates positions smoothly and accurately

You see this as:
- Drones accelerating from rest toward targets
- Smooth deceleration as they approach
- No jitter or oscillation at final position
- Natural collision avoidance during flight

---

## Lines 575-588: `Swarm.simulate(self, targets, frames)`

### Purpose:
Run simulation for a static target (one shape).

```python
self.tgt = greedy_assignment(self.pos, targets)
```
- Assign each drone to a target position

```python
for _ in range(frames):
    self.step()
    result.append(self.pos.copy())
```
- Run RK4 for specified number of frames
- Record positions for video

```python
for _ in range(HOLD_FRAMES):
    self.step()
    result.append(self.pos.copy())
```
- Extra frames at end for drones to settle
- Gives "pause" effect when shape is complete

### Video Effect:
- Transition: 90 frames of drones flying to new shape
- Hold: 20 frames of the completed shape before next transition

---

## Lines 590-600: `Swarm.track(self, target_seq, steps=5)`

### Purpose:
Track a sequence of moving targets (animated GIF).

```python
for t in target_seq:
    self.tgt = greedy_assignment(self.pos, t)
    for _ in range(steps):
        self.step()
        result.append(self.pos.copy())
```
- For each GIF frame, update target positions
- Run 5 simulation steps between each target update
- This creates smooth interpolation between GIF frames

### Video Effect:
The running tiger animation - drones continuously chase the moving shape, creating fluid motion rather than teleporting between frames.

---

# 🔵 PART 7: VIDEO GENERATION

## Lines 603-621: `render_video(frames, path, fps=30)`

### Purpose:
Convert simulation frames to MP4 video.

```python
for i, pos in enumerate(frames):
    frame = np.zeros((H, W, 3), np.uint8)
```
- Create black canvas for each frame

```python
    for p in pos:
        cv2.circle(frame, (x, y), 3, (40, 80, 80), -1)  # Outer glow
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)  # Bright center
```
- Draw each drone as a yellow circle with dark outline
- Creates "LED light" effect

### Video Effect:
Produces the final MP4 showing all drone movements with proper timing (30 fps).

---

# 🔵 SUMMARY: HOW IT ALL CONNECTS

## The Complete Pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT PROCESSING                             │
├─────────────────────────────────────────────────────────────────────┤
│ Image/GIF → Gaussian Blur → Sobel Gradients → NMS → Hysteresis     │
│                           OR                                        │
│ Image → Otsu Threshold → Binary → Edge Points                       │
│                           ↓                                         │
│               Target Positions (where drones should go)             │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        ASSIGNMENT                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Current Drone Positions + Target Positions                          │
│                    ↓                                                │
│ Euclidean Distance Matrix (all pairwise distances)                  │
│                    ↓                                                │
│ Greedy Assignment (each drone → nearest free target)                │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    PHYSICS SIMULATION                               │
├─────────────────────────────────────────────────────────────────────┤
│ For each time step:                                                 │
│   1. Spatial Hashing → Find nearby drone pairs                      │
│   2. Repulsion Forces → Push away if too close                      │
│   3. Spring Forces → Pull toward target                             │
│   4. Damping → Prevent oscillation                                  │
│   5. RK4 Integration → Update positions & velocities                │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       VIDEO OUTPUT                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Draw drones as circles on black background                          │
│ Add phase labels                                                    │
│ Encode to MP4 at 30fps                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Physical Behavior Summary:

| What You See | Mathematical Cause |
|-------------|-------------------|
| Drones start in grid | Initial condition in `__init__` |
| Drones accelerate toward target | Spring force: $k_p(\vec{T} - \vec{x})$ |
| Drones slow down near target | Damping: $-k_d\vec{v}$ |
| Drones don't collide | Repulsion: $k_{rep}/d^3$ |
| Drones don't teleport | Velocity saturation: $v_{max}$ cap |
| Smooth motion | RK4 4th-order accuracy |
| Shape is recognizable | Canny edge detection |
| Animation is fluid | Continuous tracking of moving targets |

---

## Error Analysis:

| Method | Local Error | Global Error |
|--------|-------------|--------------|
| Euler (1st order) | $O(h^2)$ | $O(h)$ |
| RK2 (2nd order) | $O(h^3)$ | $O(h^2)$ |
| **RK4 (4th order)** | $O(h^5)$ | $O(h^4)$ |

With $h = 0.05$ (DT):
- Euler error: ~5%
- RK4 error: ~0.00000625%

This is why RK4 is preferred for smooth, accurate drone trajectories.
