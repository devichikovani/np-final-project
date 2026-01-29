# Ultra-Detailed Line-by-Line Explanation

## Every Single Line of `drone_show_math.py` Explained

---

# SECTION 1: IMPORTS AND CONFIGURATION (Lines 1-34)

---

## Lines 1-11: Module Docstring

```python
"""
Illuminated Drone Show Simulation - NP 2025 Final Project
Ramaz Botchorishvili, Kutaisi International University

IVP Model: dx/dt = v*min(1, vmax/|v|), dv/dt = (1/m)[kp(T-x) + frep - kd*v]
Solved with RK4, collision avoidance via repulsive forces.

*** PURE MATHEMATICAL IMPLEMENTATION ***
All algorithms implemented from scratch using only numpy for array operations.
No scipy or cv2 algorithmic functions used.
"""
```

**What it describes:**
- This is an Initial Value Problem (IVP) simulation
- Two coupled differential equations for position and velocity
- RK4 = Runge-Kutta 4th order numerical solver
- "Pure mathematical" = no black-box library functions for core algorithms

---

## Lines 12-15: Imports

```python
import numpy as np
```
- NumPy: Numerical Python library for fast array operations
- All vectors (positions, velocities) stored as numpy arrays
- Matrix operations (distance computation) use numpy

```python
import cv2  # Only for image I/O and video writing
```
- OpenCV: Used ONLY for:
  - Reading images (`cv2.imread`)
  - Writing video frames (`cv2.VideoWriter`)
  - Drawing circles/text (`cv2.circle`, `cv2.putText`)
- NOT used for mathematical algorithms

```python
from PIL import Image
```
- Python Imaging Library
- Used only for reading GIF animation frames

```python
import os
```
- Operating system interface
- Used for file path operations (`os.path.join`, `os.path.exists`)

---

## Line 17: Random Seed

```python
np.random.seed(42)
```

**What it does:**
- Sets the random number generator to a fixed starting state
- `42` is arbitrary (it's a joke reference to "Hitchhiker's Guide to the Galaxy")

**Why it matters:**
- Makes the simulation reproducible
- Running the code twice gives identical results
- Important for debugging and consistent demonstrations

**Video effect:**
- Without this, drone initial positions might vary slightly each run
- With this, every run produces identical video output

---

## Lines 19-27: Physics Constants

```python
M = 1.0           # Mass
```
- Each drone has mass = 1 kg (arbitrary units)
- Appears in Newton's 2nd law: $F = ma$ → $a = F/m$
- Higher mass = slower acceleration for same force
- **Video:** Heavier drones would move more sluggishly

```python
V_MAX = 100.0     # Max velocity
```
- Maximum speed in pixels per second
- Velocity saturation: if $|v| > 100$, scale down
- **Video:** Prevents drones from moving faster than 100 px/s (no blur/teleporting)

```python
K_P = 25.0        # Position gain (spring constant)
```
- Spring stiffness in the spring-mass-damper model
- Force toward target: $F = k_p \cdot (target - position)$
- Higher $k_p$ = stronger pull toward target
- **Video:** Higher value = drones snap to position faster, lower = slow drift

```python
K_D = 12.0        # Damping coefficient
```
- Friction/drag coefficient
- Damping force: $F = -k_d \cdot velocity$
- Opposes motion, prevents oscillation
- **Video:** Without damping, drones would oscillate around targets forever

```python
K_REP = 50.0      # Repulsion strength
```
- Strength of collision avoidance force
- Repulsion: $F = k_{rep} / distance^2$
- **Video:** Drones push away from each other when too close

```python
R_SAFE = 4.0      # Safety radius for collision avoidance
```
- Distance threshold in pixels
- If two drones are closer than 4 pixels, they repel
- **Video:** Determines minimum spacing between drones

```python
DT = 0.05         # Time step for integration
```
- $\Delta t$ = 0.05 seconds per simulation step
- At 30 fps video, each frame = ~0.033s, so simulation runs faster than real-time
- Smaller DT = more accurate but slower simulation
- **Video:** Controls smoothness of motion

```python
W, H = 800, 600   # Canvas size
```
- Video resolution: 800 pixels wide × 600 pixels tall
- All drone positions bounded to this area
- **Video:** The visible frame dimensions

---

## Lines 29-31: Animation Settings

```python
TRANSITION_FRAMES = 90
```
- Number of frames for each shape transition
- At 30 fps: 90 frames = 3 seconds of motion
- **Video:** How long drones fly from one shape to next

```python
HOLD_FRAMES = 20
```
- Extra frames to hold the completed shape
- At 30 fps: 20 frames = 0.67 seconds pause
- **Video:** Brief pause to admire the completed formation

---

# SECTION 2: EUCLIDEAN DISTANCE MATRIX (Lines 39-61)

---

```python
def euclidean_distance_matrix(A, B):
```
- Function to compute all pairwise distances between two sets of points
- Input A: array of shape (n, 2) - n drone positions
- Input B: array of shape (m, 2) - m target positions
- Output: matrix of shape (n, m) where entry [i,j] = distance from drone i to target j

```python
    """
    Compute pairwise Euclidean distances between points in A and B.
    
    Mathematical formula: d(a,b) = sqrt(sum((a_i - b_i)^2))
    
    Optimized using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    Replaces: scipy.spatial.distance.cdist
    """
```

---

```python
    # ||a||^2 for each row in A
    A_sq = np.sum(A**2, axis=1, keepdims=True)
```

**Step-by-step:**
1. `A**2` - Square every element: if A = [[3,4], [1,2]], then A² = [[9,16], [1,4]]
2. `np.sum(..., axis=1)` - Sum across columns (axis 1): [9+16, 1+4] = [25, 5]
3. `keepdims=True` - Keep as column vector: [[25], [5]]

**Mathematical meaning:**
- For each point $\vec{a} = (x, y)$
- Computes $\|\vec{a}\|^2 = x^2 + y^2$ (squared magnitude)

---

```python
    # ||b||^2 for each row in B
    B_sq = np.sum(B**2, axis=1, keepdims=True)
```

Same computation for target positions.

---

```python
    # Dot product A @ B^T
    AB = A @ B.T
```

**Step-by-step:**
1. `B.T` - Transpose B: if B is (m, 2), B.T is (2, m)
2. `A @ B.T` - Matrix multiplication: (n, 2) × (2, m) = (n, m)
3. Each entry $AB[i,j] = A[i] \cdot B[j] = x_i \cdot x_j + y_i \cdot y_j$

**Mathematical meaning:**
- Computes all pairwise dot products
- $\vec{a} \cdot \vec{b} = x_a x_b + y_a y_b$

---

```python
    # Distance matrix: sqrt(||a||^2 + ||b||^2 - 2*a·b)
    dist_sq = A_sq + B_sq.T - 2 * AB
```

**Step-by-step:**
1. `B_sq.T` - Transpose to row vector: [[b1², b2², ...]]
2. `A_sq + B_sq.T` - Broadcasting: adds column to each row
3. `- 2 * AB` - Subtract twice the dot products

**Mathematical derivation:**
$$\|\vec{a} - \vec{b}\|^2 = (\vec{a} - \vec{b}) \cdot (\vec{a} - \vec{b})$$
$$= \vec{a} \cdot \vec{a} - 2\vec{a} \cdot \vec{b} + \vec{b} \cdot \vec{b}$$
$$= \|\vec{a}\|^2 + \|\vec{b}\|^2 - 2(\vec{a} \cdot \vec{b})$$

---

```python
    # Numerical stability: clamp negative values to 0
    dist_sq = np.maximum(dist_sq, 0)
```

**Why needed:**
- Due to floating-point rounding errors, $\|\vec{a}\|^2 + \|\vec{b}\|^2 - 2(\vec{a} \cdot \vec{b})$ might be tiny negative
- Example: mathematically 0, but computed as -0.0000000001
- `np.sqrt` of negative = NaN (Not a Number)
- `np.maximum(x, 0)` clamps negatives to exactly 0

---

```python
    return np.sqrt(dist_sq)
```

**What it does:**
- Takes square root of each element
- Converts squared distances to actual distances

**Video effect:**
- This matrix tells us how far each drone is from each target
- Used by assignment algorithm to decide which drone goes where
- Closer drone-target pairs create shorter, more efficient paths

---

# SECTION 3: SPATIAL HASH GRID (Lines 64-114)

---

```python
def spatial_hash_grid(positions, cell_size):
```
- Divides 2D space into a grid of square cells
- Each cell tracks which drones are inside it
- Enables fast neighbor lookup

```python
    """
    Build a spatial hash grid for O(n) neighbor queries.
    
    Mathematical concept: Discretize continuous space into cells.
    Hash function: h(x,y) = (floor(x/cell_size), floor(y/cell_size))
    
    Replaces: scipy.spatial.cKDTree
    """
```

---

```python
    cells = {}
```
- Empty dictionary to store cell → drone mapping
- Key: (cell_x, cell_y) tuple
- Value: list of drone indices in that cell

---

```python
    cell_indices = (positions / cell_size).astype(int)
```

**Step-by-step:**
1. `positions / cell_size` - Divide all coordinates by cell size
   - If position = (17.3, 25.8) and cell_size = 4
   - Result = (4.325, 6.45)
2. `.astype(int)` - Truncate to integer (floor)
   - Result = (4, 6)

**Mathematical meaning:**
- Hash function: $h(x, y) = (\lfloor x/s \rfloor, \lfloor y/s \rfloor)$
- Maps continuous coordinates to discrete grid cells

---

```python
    for i, (cx, cy) in enumerate(cell_indices):
```
- Loop through all drones
- `i` = drone index (0, 1, 2, ...)
- `(cx, cy)` = cell coordinates for drone i

---

```python
        key = (cx, cy)
        if key not in cells:
            cells[key] = []
        cells[key].append(i)
```

**What it does:**
- Create cell if it doesn't exist
- Add drone index to that cell's list

**Example:**
- Drone 0 at cell (4, 6)
- Drone 5 at cell (4, 6)
- cells = {(4, 6): [0, 5], ...}

---

```python
    return cells, cell_indices
```
- Returns the dictionary and the computed cell indices
- Both used by `find_neighbor_pairs`

---

## Lines 85-114: Finding Neighbor Pairs

```python
def find_neighbor_pairs(positions, radius):
```
- Finds all pairs of drones within `radius` of each other
- Uses spatial hashing for efficiency

---

```python
    cell_size = radius
    cells, cell_indices = spatial_hash_grid(positions, cell_size)
```
- Cell size = collision radius
- This ensures neighbors can only be in same cell or adjacent cells

---

```python
    pairs = []
    n = len(positions)
    checked = set()
```
- `pairs` - List to store (i, j) pairs of colliding drones
- `checked` - Set to avoid checking same pair twice

---

```python
    for i in range(n):
        cx, cy = cell_indices[i]
```
- For each drone i, get its cell coordinates

---

```python
        # Check 3x3 neighborhood of cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
```

**Why 3×3 neighborhood?**
- A drone at the edge of cell (4, 6) might be within radius of a drone in cell (5, 6)
- Must check all 9 neighboring cells (including own cell)

```
┌─────┬─────┬─────┐
│(3,5)│(4,5)│(5,5)│
├─────┼─────┼─────┤
│(3,6)│(4,6)│(5,6)│  ← drone i is in (4,6)
├─────┼─────┼─────┤
│(3,7)│(4,7)│(5,7)│
└─────┴─────┴─────┘
```

---

```python
                if key in cells:
                    for j in cells[key]:
                        if j > i:  # Avoid duplicate pairs
```

**Why `j > i`?**
- Without this, we'd check both (0,5) and (5,0)
- Only checking j > i ensures each pair is found exactly once

---

```python
                            pair_key = (i, j)
                            if pair_key not in checked:
                                checked.add(pair_key)
```
- Extra safety to avoid duplicates (when drone appears in multiple neighbor checks)

---

```python
                                # Euclidean distance check
                                diff = positions[i] - positions[j]
                                dist = np.sqrt(diff[0]**2 + diff[1]**2)
                                if dist < radius:
                                    pairs.append((i, j))
```

**What it does:**
1. Compute displacement: $\vec{d} = \vec{x}_i - \vec{x}_j$
2. Compute distance: $d = \sqrt{d_x^2 + d_y^2}$
3. If within radius, add to pairs list

**Video effect:**
- These pairs are the drones that will push away from each other
- Creates the "collision avoidance" behavior visible when drones get close

---

# SECTION 4: GREEDY ASSIGNMENT (Lines 117-140)

---

```python
def greedy_assignment(positions, targets):
```
- Assigns each drone to a unique target position
- Uses greedy strategy: each drone picks nearest available target

---

```python
    n = len(positions)
    dist = euclidean_distance_matrix(positions, targets)
```
- `n` = number of drones
- `dist` = n×n matrix of all drone-to-target distances

---

```python
    result = np.zeros((n, 2))
    assigned_targets = np.zeros(n, dtype=bool)
```
- `result` = array to store assigned target for each drone
- `assigned_targets` = boolean array tracking which targets are taken

---

```python
    # Process each drone
    for i in range(n):
```
- Process drones in order (0, 1, 2, ...)

---

```python
        # Mask already assigned targets with infinity
        masked_dist = dist[i].copy()
        masked_dist[assigned_targets] = np.inf
```

**What it does:**
1. Copy row i of distance matrix (distances from drone i to all targets)
2. Set distance to infinity for already-taken targets
3. This prevents selecting an occupied target

---

```python
        # Find nearest available target
        j = np.argmin(masked_dist)
```
- `np.argmin` returns index of minimum value
- With infinities masking taken targets, finds nearest FREE target

---

```python
        result[i] = targets[j]
        assigned_targets[j] = True
```
- Record assignment: drone i → target j
- Mark target j as taken

---

```python
    return result
```
- Returns array of assigned targets (same order as drones)

**Video effect:**
- Good assignment = short paths, no crossing
- Drones flow smoothly from old shape to new shape
- Bad assignment would cause chaotic crossing paths

---

# SECTION 5: GAUSSIAN KERNEL (Lines 143-157)

---

```python
def gaussian_kernel_2d(size, sigma):
```
- Creates a 2D Gaussian blur kernel
- Used for noise reduction before edge detection

---

```python
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
```

**Example for size=5:**
- `-5//2 + 1 = -1`
- `5//2 + 1 = 3`
- `ax = [-1, 0, 1, 2]` (approximately centered)

For size=3: `ax = [0, 1]` → center not perfectly at 0

---

```python
    xx, yy = np.meshgrid(ax, ax)
```

**What `meshgrid` does:**
- Creates 2D coordinate arrays from 1D arrays
```
ax = [-1, 0, 1]
xx = [[-1, 0, 1],    yy = [[-1, -1, -1],
      [-1, 0, 1],          [ 0,  0,  0],
      [-1, 0, 1]]          [ 1,  1,  1]]
```
- xx[i,j] = x-coordinate, yy[i,j] = y-coordinate

---

```python
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
```

**Mathematical formula:**
$$G(x, y) = e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

This is the un-normalized Gaussian function:
- Maximum at center (0, 0) where $e^0 = 1$
- Decreases exponentially with distance from center
- $\sigma$ controls the spread (blur amount)

---

```python
    return kernel / kernel.sum()  # Normalize
```

**Why normalize?**
- Without normalization, blurring would brighten the image
- Dividing by sum ensures weights add to 1
- Preserves average brightness

**Example 3×3 kernel (σ=1):**
```
[ 0.075  0.124  0.075 ]
[ 0.124  0.204  0.124 ]   Sum = 1.0
[ 0.075  0.124  0.075 ]
```

---

# SECTION 6: 2D CONVOLUTION (Lines 160-186)

---

```python
def convolve_2d(image, kernel):
```
- Applies a filter (kernel) to an image
- Core operation for all image processing

---

```python
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
```
- `kh, kw` = kernel height and width
- `pad_h, pad_w` = how much to pad image on each side

---

```python
    # Pad image
    padded = np.pad(image.astype(float), ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
```

**What padding does:**
- Adds extra pixels around image border
- `mode='edge'` copies edge pixels outward
- Without padding, output would be smaller than input

**Example (pad=1):**
```
Original:        Padded:
[1 2 3]         [1 1 2 3 3]
[4 5 6]    →    [1 1 2 3 3]
[7 8 9]         [4 4 5 6 6]
                [7 7 8 9 9]
                [7 7 8 9 9]
```

---

```python
    h, w = image.shape
    
    # Vectorized convolution using numpy broadcasting
    output = np.zeros((h, w))
    for i in range(kh):
        for j in range(kw):
            output += padded[i:i+h, j:j+w] * kernel[i, j]
```

**What this loop does:**
- For each kernel position (i, j):
  - Shift the image by (i, j)
  - Multiply by kernel weight at (i, j)
  - Add to output

**Mathematical formula:**
$$(f * g)(x, y) = \sum_{i} \sum_{j} f(i, j) \cdot g(x-i, y-j)$$

**Visual explanation for 3×3 kernel:**
```
Kernel:           For pixel output[x,y]:
[a b c]           output[x,y] = a*img[x-1,y-1] + b*img[x-1,y] + c*img[x-1,y+1]
[d e f]                       + d*img[x,y-1]   + e*img[x,y]   + f*img[x,y+1]
[g h i]                       + g*img[x+1,y-1] + h*img[x+1,y] + i*img[x+1,y+1]
```

---

```python
    return output
```

**Video effect:**
- This is used for blurring (with Gaussian kernel)
- Also used for edge detection (with Sobel kernels)
- Smooth blur → less noise → cleaner drone formations

---

# SECTION 7: SOBEL GRADIENTS (Lines 204-234)

---

```python
def sobel_gradients(image):
```
- Computes image gradients (rate of change in intensity)
- Edges = places where intensity changes rapidly

---

```python
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
```

**What Sobel-X detects:**
- Vertical edges (where left differs from right)
- Example response:
```
Image:           Gx result:
[0 0 255]        [high]  (big difference left-right)
[0 0 255]   →    [high]
[0 0 255]        [high]
```

**Mathematical meaning:**
- Approximates $\frac{\partial I}{\partial x}$ (horizontal derivative)
- Weights: center row weighted 2x for smoothing

---

```python
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=float)
```

**What Sobel-Y detects:**
- Horizontal edges (where top differs from bottom)
- Approximates $\frac{\partial I}{\partial y}$ (vertical derivative)

---

```python
    gx = convolve_2d(image, sobel_x)
    gy = convolve_2d(image, sobel_y)
```
- Apply both kernels to get x and y gradients

---

```python
    magnitude = np.sqrt(gx**2 + gy**2)
```

**Mathematical formula:**
$$|\nabla I| = \sqrt{G_x^2 + G_y^2}$$

- Gradient magnitude = edge strength
- High magnitude = strong edge
- Low magnitude = flat area

---

```python
    direction = np.arctan2(gy, gx)
```

**Mathematical formula:**
$$\theta = \arctan\left(\frac{G_y}{G_x}\right)$$

- Direction perpendicular to edge
- Used by non-maximum suppression to thin edges

---

```python
    return magnitude, direction
```

**Video effect:**
- Magnitude image shows all edges (outlines of shapes)
- These become target positions for drones
- Drones arrange themselves along detected edges

---

# SECTION 8: NON-MAXIMUM SUPPRESSION (Lines 237-273)

---

```python
def non_maximum_suppression(magnitude, direction):
```
- Thins thick edges to 1-pixel width
- Keeps only local maxima in gradient direction

---

```python
    h, w = magnitude.shape
    output = np.zeros((h, w))
```
- Create empty output array

---

```python
    # Convert angle to degrees and map to 0-180
    angle = direction * 180 / np.pi
    angle[angle < 0] += 180
```

**What it does:**
1. Convert radians to degrees: $\theta_{deg} = \theta_{rad} \cdot \frac{180}{\pi}$
2. Map negative angles to positive (−45° → 135°)
3. Result: angles in range [0, 180)

---

```python
    # Pad magnitude for neighbor access
    mag_pad = np.pad(magnitude, 1, mode='constant', constant_values=0)
```
- Add 1-pixel border of zeros
- Allows safe neighbor access at image edges

---

```python
    # Vectorized comparison using angle bins
    angle_bin = np.zeros((h, w), dtype=int)
    angle_bin[(angle >= 22.5) & (angle < 67.5)] = 1
    angle_bin[(angle >= 67.5) & (angle < 112.5)] = 2
    angle_bin[(angle >= 112.5) & (angle < 157.5)] = 3
```

**Angle binning:**
| Bin | Angle Range | Direction | Check Neighbors |
|-----|-------------|-----------|-----------------|
| 0   | 0-22.5° or 157.5-180° | Horizontal | Left & Right |
| 1   | 22.5-67.5° | Diagonal / | Top-right & Bottom-left |
| 2   | 67.5-112.5° | Vertical | Top & Bottom |
| 3   | 112.5-157.5° | Diagonal \ | Top-left & Bottom-right |

---

```python
    for i in range(1, h+1):
        for j in range(1, w+1):
            ab = angle_bin[i-1, j-1]
            m = mag_pad[i, j]
```
- Loop through all pixels
- Get angle bin and magnitude

---

```python
            if ab == 0:  # Horizontal edge
                n1, n2 = mag_pad[i, j-1], mag_pad[i, j+1]
            elif ab == 1:  # Diagonal /
                n1, n2 = mag_pad[i-1, j+1], mag_pad[i+1, j-1]
            elif ab == 2:  # Vertical edge
                n1, n2 = mag_pad[i-1, j], mag_pad[i+1, j]
            else:  # Diagonal \
                n1, n2 = mag_pad[i-1, j-1], mag_pad[i+1, j+1]
```

**Neighbor selection based on gradient direction:**
```
For horizontal edge (gradient points left-right):
    Check left and right neighbors
    
For vertical edge (gradient points up-down):
    Check top and bottom neighbors
```

---

```python
            if m >= n1 and m >= n2:
                output[i-1, j-1] = m
```

**The suppression rule:**
- Keep pixel only if it's ≥ both neighbors in gradient direction
- Otherwise suppress (leave as 0)

**Visual effect:**
```
Before NMS:          After NMS:
[50  80  90  70  40]    [0  0 90  0  0]
[60 100 120  90  50] →  [0  0 120 0  0]
[40  70  80  60  30]    [0  0 80  0  0]
```
Thick edge becomes thin 1-pixel line.

---

# SECTION 9: HYSTERESIS THRESHOLDING (Lines 276-308)

---

```python
def hysteresis_threshold(image, low_thresh, high_thresh):
```
- Two-threshold edge detection
- Connects weak edges to strong edges

---

```python
    strong = 255
    weak = 50
```
- Strong edges marked as 255 (white)
- Weak edges temporarily marked as 50

---

```python
    strong_i, strong_j = np.where(image >= high_thresh)
    weak_i, weak_j = np.where((image >= low_thresh) & (image < high_thresh))
    
    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak
```

**Three categories:**
1. Strong edges: magnitude ≥ high_thresh → definitely edges
2. Weak edges: low_thresh ≤ magnitude < high_thresh → maybe edges
3. Non-edges: magnitude < low_thresh → definitely not edges

---

```python
    # Connect weak edges to strong edges (simple 8-connectivity)
    for i in range(1, h-1):
        for j in range(1, w-1):
            if output[i, j] == weak:
                # Check 8-connected neighbors for strong edge
                neighbors = output[i-1:i+2, j-1:j+2]
                if np.any(neighbors == strong):
                    output[i, j] = strong
                else:
                    output[i, j] = 0
```

**Logic:**
- Weak edge connected to strong edge → promote to strong
- Weak edge not connected → suppress to 0

**Why this helps:**
- Noise creates isolated weak edges → removed
- Real edges have connected weak parts → kept

---

```python
    # Remove remaining weak edges
    output[output == weak] = 0
```
- Any weak edges not promoted are discarded

---

# SECTION 10: CANNY EDGE DETECTION (Lines 311-332)

---

```python
def canny_edge_detection(image, low_thresh=50, high_thresh=150):
```
- Complete edge detection pipeline
- Combines all previous steps

---

```python
    # Step 1: Gaussian blur
    blurred = gaussian_blur(image, kernel_size=5, sigma=1.4)
```
- Reduce noise before edge detection
- 5×5 kernel with σ=1.4 is standard

---

```python
    # Step 2: Compute gradients
    magnitude, direction = sobel_gradients(blurred)
```
- Find edges using Sobel operators

---

```python
    # Normalize magnitude to 0-255
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
```
- Scale to 8-bit range for thresholding

---

```python
    # Step 3: Non-maximum suppression
    thin_edges = non_maximum_suppression(magnitude.astype(float), direction)
```
- Thin edges to 1 pixel width

---

```python
    # Step 4: Hysteresis thresholding
    edges = hysteresis_threshold(thin_edges, low_thresh, high_thresh)
    
    return edges
```
- Connect and filter edges

**Video effect:**
- Input: photograph of tiger
- Output: clean outline of tiger shape
- Drones position themselves on this outline

---

# SECTION 11: OTSU THRESHOLDING (Lines 335-364)

---

```python
def otsu_threshold(image):
```
- Automatically finds optimal threshold
- Separates foreground from background

---

```python
    # Compute histogram (256 bins for 8-bit image)
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()
```

**What `histogram` does:**
- Counts pixels at each intensity level 0-255
- Normalize to probabilities (sum = 1)

---

```python
    best_thresh = 0
    best_variance = 0
    
    for t in range(1, 256):
```
- Try every possible threshold (1 to 255)
- Find the one that best separates classes

---

```python
        # Class probabilities
        w0 = hist[:t].sum()   # P(pixel ≤ t)
        w1 = hist[t:].sum()   # P(pixel > t)
```
- $\omega_0$ = probability of background (dark pixels)
- $\omega_1$ = probability of foreground (bright pixels)

---

```python
        if w0 == 0 or w1 == 0:
            continue
```
- Skip if one class is empty (can't compute mean)

---

```python
        # Class means
        mu0 = np.sum(np.arange(t) * hist[:t]) / w0
        mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
```

**Mathematical formulas:**
$$\mu_0 = \frac{\sum_{i=0}^{t-1} i \cdot p(i)}{\omega_0}$$
$$\mu_1 = \frac{\sum_{i=t}^{255} i \cdot p(i)}{\omega_1}$$

- Mean intensity of each class

---

```python
        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1)**2
```

**Otsu's criterion:**
$$\sigma_B^2(t) = \omega_0 \omega_1 (\mu_0 - \mu_1)^2$$

- Measures separation between classes
- Larger variance = better separation

---

```python
        if variance > best_variance:
            best_variance = variance
            best_thresh = t
```
- Track the threshold with maximum variance

---

```python
    # Apply threshold
    binary = np.where(image > best_thresh, 0, 255).astype(np.uint8)
    return best_thresh, binary
```
- Create binary image using optimal threshold
- Note: inverted (dark → white, bright → black) for our use case

**Video effect:**
- Fallback when edge detection gives sparse points
- Fills in shape interior with drone positions

---

# SECTION 12: SWARM CLASS - INITIALIZATION (Lines 479-503)

---

```python
class Swarm:
    """
    Drone swarm with IVP physics model and RK4 integration.
    """
```
- Main simulation class
- Holds positions, velocities, and targets for all drones

---

```python
    def __init__(self, n):
        self.n = n
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.tgt = np.zeros((n, 2))
```

**State variables:**
- `n` = number of drones
- `pos` = position array, shape (n, 2): [x, y] for each drone
- `vel` = velocity array, shape (n, 2): [vx, vy] for each drone
- `tgt` = target array, shape (n, 2): where each drone should go

---

```python
        # Initialize on grid
        cols = int(np.sqrt(n * W / H))
        rows = int(np.ceil(n / cols))
```

**Grid calculation:**
- For 1200 drones on 800×600 canvas:
- Aspect ratio = 800/600 = 1.33
- cols = √(1200 × 1.33) ≈ 40
- rows = ⌈1200/40⌉ = 30

This creates a grid matching the canvas aspect ratio.

---

```python
        xx, yy = np.meshgrid(np.linspace(80, W-80, cols), np.linspace(80, H-80, rows))
        grid = np.column_stack([xx.ravel(), yy.ravel()])[:n]
```

**What this creates:**
1. `linspace(80, 720, 40)` - 40 evenly spaced x-coordinates
2. `linspace(80, 520, 30)` - 30 evenly spaced y-coordinates
3. `meshgrid` - creates 2D grid of all combinations
4. `column_stack + ravel` - flatten to list of (x, y) points
5. `[:n]` - take only first n points

---

```python
        if len(grid) < n:
            grid = np.vstack([grid, np.random.rand(n - len(grid), 2) * [W-160, H-160] + 80])
        self.pos = grid.copy()
```
- If not enough grid points, add random ones
- Copy to position array

**Video effect:**
- At video start, drones appear in neat rectangular grid
- This is the initial formation before any animation

---

# SECTION 13: REPULSION FORCE (Lines 505-527)

---

```python
    def repulsion(self):
```
- Computes collision avoidance forces

---

```python
        forces = np.zeros((self.n, 2))
```
- Initialize force array to zero

---

```python
        # Use spatial hashing instead of KD-Tree
        pairs = find_neighbor_pairs(self.pos, R_SAFE)
        
        if not pairs:
            return forces
```
- Find all drone pairs within R_SAFE = 4 pixels
- If none, return zero forces

---

```python
        pairs = np.array(pairs)
        diff = self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]]
```

**What this computes:**
- `pairs[:, 0]` - indices of first drone in each pair
- `pairs[:, 1]` - indices of second drone in each pair
- `diff` - displacement vectors: $\vec{x}_i - \vec{x}_j$

---

```python
        # Euclidean distance: ||xi - xj||
        dist = np.sqrt(np.sum(diff**2, axis=1))
        dist = np.maximum(dist, 0.1)  # Avoid division by zero
```
- Compute distance for each pair
- Clamp minimum to 0.1 to avoid infinity

---

```python
        # Force magnitude: K_REP / d^3, direction: unit vector * d = diff/d
        # So force = K_REP * diff / d^3
        fvec = (K_REP / dist**3)[:, None] * diff
```

**Mathematical formula:**
$$\vec{F}_{rep} = k_{rep} \cdot \frac{\vec{x}_i - \vec{x}_j}{|\vec{x}_i - \vec{x}_j|^3}$$

**Breaking it down:**
- Direction: $\frac{\vec{x}_i - \vec{x}_j}{|\vec{x}_i - \vec{x}_j|}$ (unit vector pointing from j to i)
- Magnitude: $\frac{k_{rep}}{|\vec{x}_i - \vec{x}_j|^2}$ (inverse square)
- Combined: $\frac{k_{rep}}{d^3} \cdot \vec{d}$

**Physical meaning:**
- Closer drones = stronger repulsion (inverse square law)
- Same principle as electrostatic/gravitational forces

---

```python
        # Accumulate forces (Newton's 3rd law: equal and opposite)
        np.add.at(forces, pairs[:, 0], fvec)
        np.add.at(forces, pairs[:, 1], -fvec)
        
        return forces
```

**Newton's Third Law:**
- Drone i feels force $+\vec{F}$ from drone j
- Drone j feels force $-\vec{F}$ from drone i
- Equal magnitude, opposite direction

**Video effect:**
- Drones push away when too close
- Creates natural spacing in formations
- Prevents overlapping/collision during transitions

---

# SECTION 14: RK4 INTEGRATION (Lines 529-573)

---

```python
    def step(self):
        """
        RK4 integration of IVP equations.
        """
```
- Advances simulation by one time step
- This is THE CORE numerical method

---

```python
        def deriv(p, v):
            """Compute derivatives dx/dt and dv/dt."""
```
- Inner function computing the right-hand side of our differential equations
- Called 4 times per RK4 step

---

```python
            # Velocity magnitude
            vnorm = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
            vnorm = np.maximum(vnorm, 1e-6)
```

**What this computes:**
- $|\vec{v}| = \sqrt{v_x^2 + v_y^2}$ for each drone
- Clamp to tiny positive value to avoid 0/0

---

```python
            # Velocity saturation: dx/dt = v * min(1, vmax/|v|)
            dx = v * np.minimum(1.0, V_MAX / vnorm)
```

**Mathematical formula:**
$$\frac{d\vec{x}}{dt} = \vec{v} \cdot \min\left(1, \frac{v_{max}}{|\vec{v}|}\right)$$

**What this does:**
- If $|\vec{v}| \leq v_{max}$: $\frac{v_{max}}{|\vec{v}|} \geq 1$, so $\min = 1$, use full velocity
- If $|\vec{v}| > v_{max}$: $\frac{v_{max}}{|\vec{v}|} < 1$, scale down velocity

**Physical meaning:**
- Maximum speed is capped at V_MAX = 100 px/s
- Prevents unrealistic speeds

---

```python
            # Compute repulsion at position p
            old_pos = self.pos.copy()
            self.pos = p
            rep = self.repulsion()
            self.pos = old_pos
```

**Why save/restore?**
- RK4 evaluates derivatives at intermediate positions
- Need to compute repulsion at those hypothetical positions
- Must restore original positions afterward

---

```python
            # dv/dt = (1/m)[kp(T-x) + frep - kd*v]
            dv = (K_P * (self.tgt - p) + rep - K_D * v) / M
            
            return dx, dv
```

**The acceleration equation:**
$$\frac{d\vec{v}}{dt} = \frac{1}{m}\left[k_p(\vec{T} - \vec{x}) + \vec{F}_{rep} - k_d\vec{v}\right]$$

**Breaking down each term:**

| Term | Formula | Physical Meaning |
|------|---------|------------------|
| Spring | $k_p(\vec{T} - \vec{x})$ | Force pulling toward target. Like a spring attached to target. |
| Repulsion | $\vec{F}_{rep}$ | Force pushing away from nearby drones |
| Damping | $-k_d\vec{v}$ | Friction opposing motion. Prevents oscillation. |
| Mass | $1/m$ | Newton's 2nd law: $a = F/m$ |

---

## RK4 Stages:

```python
        # RK4 stages
        k1x, k1v = deriv(self.pos, self.vel)
```
**Stage 1:** Derivative at current state
- $k_1 = f(t_n, y_n)$

```python
        k2x, k2v = deriv(self.pos + 0.5*DT*k1x, self.vel + 0.5*DT*k1v)
```
**Stage 2:** Derivative at midpoint using k1
- $k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1)$

```python
        k3x, k3v = deriv(self.pos + 0.5*DT*k2x, self.vel + 0.5*DT*k2v)
```
**Stage 3:** Derivative at midpoint using k2
- $k_3 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_2)$

```python
        k4x, k4v = deriv(self.pos + DT*k3x, self.vel + DT*k3v)
```
**Stage 4:** Derivative at endpoint using k3
- $k_4 = f(t_n + h, y_n + hk_3)$

---

## RK4 Update:

```python
        # Update state
        self.pos += (DT/6) * (k1x + 2*k2x + 2*k3x + k4x)
        self.vel += (DT/6) * (k1v + 2*k2v + 2*k3v + k4v)
```

**Mathematical formula:**
$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**Why these weights (1, 2, 2, 1)?**
- Derived from Simpson's rule for integration
- Middle samples (k2, k3) weighted double
- Achieves 4th-order accuracy: local error ~ $O(h^5)$

**Comparison with simpler methods:**

| Method | Formula | Local Error | Global Error |
|--------|---------|-------------|--------------|
| Euler | $y_{n+1} = y_n + hf(t_n, y_n)$ | $O(h^2)$ | $O(h)$ |
| Midpoint (RK2) | Uses k1, k2 | $O(h^3)$ | $O(h^2)$ |
| **RK4** | Uses k1, k2, k3, k4 | $O(h^5)$ | $O(h^4)$ |

With h = 0.05:
- Euler: ~5% error
- RK4: ~0.000006% error

---

```python
        # Boundary constraints
        self.pos = np.clip(self.pos, 5, [W-5, H-5])
```
- Keep drones inside canvas
- 5-pixel margin from edges

---

# SECTION 15: SIMULATION METHODS (Lines 575-600)

---

```python
    def simulate(self, targets, frames):
        """Simulate transition to static targets."""
        self.tgt = greedy_assignment(self.pos, targets)
```
- Assign each drone to a target using greedy algorithm

---

```python
        result = []
        for _ in range(frames):
            self.step()
            result.append(self.pos.copy())
```
- Run simulation for specified frames
- Save position snapshot each frame

---

```python
        for _ in range(HOLD_FRAMES):
            self.step()
            result.append(self.pos.copy())
        return result
```
- Extra frames for drones to settle
- Creates "pause" effect at each shape

**Video effect:**
- Drones fly from current positions to new shape
- Hold briefly before next transition

---

```python
    def track(self, target_seq, steps=5):
        """Simulate dynamic tracking of moving targets."""
        result = []
        for t in target_seq:
            self.tgt = greedy_assignment(self.pos, t)
            for _ in range(steps):
                self.step()
                result.append(self.pos.copy())
        return result
```

**For animated GIF:**
- `target_seq` = list of target positions (one per GIF frame)
- For each new target frame:
  - Reassign drones to new positions
  - Run 5 simulation steps
  - Creates smooth interpolation between frames

**Video effect:**
- Running tiger animation
- Drones continuously chase the moving shape
- Fluid motion instead of teleporting

---

# SECTION 16: VIDEO RENDERING (Lines 603-621)

---

```python
def render_video(frames, path, fps=30, label_override=None):
```
- Converts simulation frames to MP4 video

---

```python
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
```
- Create video file
- 'mp4v' codec for MP4 format
- 30 fps, 800×600 resolution

---

```python
    for i, pos in enumerate(frames):
        frame = np.zeros((H, W, 3), np.uint8)
```
- Create black canvas (H×W×3 for RGB)

---

```python
        for p in pos:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(frame, (x, y), 3, (40, 80, 80), -1)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
```

**Drawing each drone:**
1. First circle: radius 3, dark color (40, 80, 80) - outer glow
2. Second circle: radius 2, bright yellow (0, 255, 255) - LED center

**Color format:** BGR (not RGB) - OpenCV convention

---

```python
        cv2.putText(frame, label, 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
    out.release()
```
- Add text label (phase indicator)
- Write frame to video
- Close video file

---

# SUMMARY: COMPLETE DATA FLOW

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              INITIALIZATION                                │
│  Swarm.__init__() creates n drones on grid                                │
│  pos = [[x1,y1], [x2,y2], ...]  vel = [[0,0], [0,0], ...]                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 1: STATIC SHAPE                           │
│  1. Load image → Canny edge detection → edge points                       │
│  2. greedy_assignment(drone_positions, edge_points)                        │
│  3. Run step() × 90 frames → drones fly to shape                          │
│  4. Run step() × 20 frames → hold position                                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                      INSIDE EACH step() CALL                               │
│                                                                            │
│  1. k1 = deriv(pos, vel)                                                  │
│     ├─ vnorm = ||vel||                                                    │
│     ├─ dx = vel × min(1, vmax/vnorm)     [velocity saturation]            │
│     ├─ rep = repulsion()                                                  │
│     │   ├─ pairs = find_neighbor_pairs()  [spatial hash]                  │
│     │   └─ force = Krep × diff / dist³   [inverse square]                 │
│     └─ dv = (Kp(tgt-pos) + rep - Kd×vel) / M                             │
│                                                                            │
│  2. k2 = deriv(pos + 0.5×DT×k1x, vel + 0.5×DT×k1v)                        │
│  3. k3 = deriv(pos + 0.5×DT×k2x, vel + 0.5×DT×k2v)                        │
│  4. k4 = deriv(pos + DT×k3x, vel + DT×k3v)                                │
│                                                                            │
│  5. pos += (DT/6)(k1x + 2k2x + 2k3x + k4x)                                │
│  6. vel += (DT/6)(k1v + 2k2v + 2k3v + k4v)                                │
│  7. pos = clip(pos, boundaries)                                           │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 3: ANIMATION                              │
│  For each GIF frame:                                                       │
│    1. Extract new target points from frame                                 │
│    2. Reassign drones to new targets                                       │
│    3. Run step() × 5 frames → smooth interpolation                        │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                            VIDEO OUTPUT                                    │
│  For each saved position frame:                                            │
│    1. Draw black canvas                                                    │
│    2. Draw each drone as glowing circle                                    │
│    3. Add phase label text                                                 │
│    4. Write to MP4 file                                                    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

# KEY EQUATIONS REFERENCE

## IVP System (State-Space Form)

State vector: $\mathbf{y} = \begin{pmatrix} \vec{x} \\ \vec{v} \end{pmatrix}$

$$\frac{d\mathbf{y}}{dt} = \begin{pmatrix} \vec{v} \cdot \min(1, v_{max}/|\vec{v}|) \\ \frac{1}{m}[k_p(\vec{T}-\vec{x}) + \vec{F}_{rep} - k_d\vec{v}] \end{pmatrix}$$

## RK4 Integration

$$\mathbf{y}_{n+1} = \mathbf{y}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$$

Where:
- $\mathbf{k}_1 = f(t_n, \mathbf{y}_n)$
- $\mathbf{k}_2 = f(t_n + \frac{\Delta t}{2}, \mathbf{y}_n + \frac{\Delta t}{2}\mathbf{k}_1)$
- $\mathbf{k}_3 = f(t_n + \frac{\Delta t}{2}, \mathbf{y}_n + \frac{\Delta t}{2}\mathbf{k}_2)$
- $\mathbf{k}_4 = f(t_n + \Delta t, \mathbf{y}_n + \Delta t \cdot \mathbf{k}_3)$

## Repulsion Force

$$\vec{F}_{rep,i} = \sum_{j: |\vec{x}_i - \vec{x}_j| < R_{safe}} k_{rep} \cdot \frac{\vec{x}_i - \vec{x}_j}{|\vec{x}_i - \vec{x}_j|^3}$$

## Canny Edge Detection

1. Gaussian blur: $G_\sigma * I$
2. Gradient: $\nabla I = (G_x * I, G_y * I)$
3. Magnitude: $|\nabla I| = \sqrt{G_x^2 + G_y^2}$
4. Non-maximum suppression
5. Hysteresis thresholding

## Otsu's Method

Optimal threshold: $t^* = \arg\max_t \omega_0(t)\omega_1(t)(\mu_0(t) - \mu_1(t))^2$
