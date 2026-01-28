# Mathematical Implementation Notes

## Drone Show Simulation - NP 2025 Final Project

This document explains which parts use pure mathematical implementations vs library functions, and the reasoning behind each choice.

---

## KEPT AS PURE MATHEMATICS (Core Algorithms)

These are the numerical methods required by the project specification:

### 1. RK4 Integration (4th-order Runge-Kutta)
**Location:** `Swarm.step()` method (lines 570-600)

**Mathematical Formula:**
```
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
k₃ = f(tₙ + h/2, yₙ + h·k₂/2)
k₄ = f(tₙ + h, yₙ + h·k₃)
yₙ₊₁ = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

**Why kept as math:** This IS the core numerical method for solving the IVP. Using `scipy.integrate.solve_ivp` would defeat the purpose of the project.

---

### 2. IVP Physics Model
**Location:** `deriv()` function inside `Swarm.step()`

**Mathematical Formulas:**
```
dx/dt = v · min(1, vₘₐₓ/|v|)     # Velocity with saturation
dv/dt = (1/m)[kₚ(T-x) + fᵣₑₚ - kᵈv]   # Spring-damper with repulsion
```

**Why kept as math:** This is the differential equation we're solving - it must be implemented explicitly.

---

### 3. Repulsive Force Calculation
**Location:** `Swarm.repulsion()` method

**Mathematical Formula:**
```
fᵣₑₚ = kᵣₑₚ · (xᵢ - xⱼ) / |xᵢ - xⱼ|³   if |xᵢ - xⱼ| < Rₛₐfₑ
```

This is an inverse-square repulsive force (similar to electrostatic/gravitational).

**Why kept as math:** Core physics model, no library equivalent.

---

### 4. Euclidean Distance Matrix
**Location:** `euclidean_distance_matrix()` function

**Mathematical Formula:**
```
d(a,b) = √(Σ(aᵢ - bᵢ)²)

Optimized: ||a-b||² = ||a||² + ||b||² - 2·a·b
```

**Why kept as math:** Simple vectorized numpy, demonstrates understanding of distance computation.

---

### 5. Spatial Hash Grid (Collision Detection)
**Location:** `spatial_hash_grid()` and `find_neighbor_pairs()` functions

**Mathematical Concept:**
```
Hash function: h(x,y) = (⌊x/cell_size⌋, ⌊y/cell_size⌋)
```

Discretizes continuous 2D space into grid cells for O(n) neighbor lookup instead of O(n²).

**Why kept as math:** Demonstrates spatial data structure understanding.

---

### 6. Greedy Assignment Algorithm
**Location:** `greedy_assignment()` function

**Algorithm:**
```
For each drone i:
    Find nearest unassigned target j
    Assign drone i → target j
    Mark target j as assigned
```

**Why kept as math:** Shows assignment algorithm understanding. Not globally optimal like Hungarian, but O(n²) vs O(n³).

---

## SWITCHED TO LIBRARY (Image I/O - Not Core Math)

These were switched from pure Python to OpenCV for **performance reasons only**:

### 1. Gaussian Blur → `cv2.GaussianBlur`

**Original Math Implementation:**
```python
def gaussian_kernel_2d(size, sigma):
    G(x,y) = (1/(2πσ²)) · exp(-(x² + y²)/(2σ²))
    
def convolve_2d(image, kernel):
    (f * g)(x,y) = Σᵢ Σⱼ f(i,j) · g(x-i, y-j)
```

**Why switched:** 
- Pure Python nested loops are ~1000x slower than C++ OpenCV
- Processing 41 GIF frames took 10+ minutes vs 2 seconds
- This is IMAGE I/O, not the core numerical method

---

### 2. Canny Edge Detection → `cv2.Canny`

**Original Math Implementation:**
```python
def canny_edge_detection():
    1. Gaussian blur (noise reduction)
    2. Sobel gradients: Gₓ, Gᵧ, magnitude = √(Gₓ² + Gᵧ²)
    3. Non-maximum suppression (edge thinning)
    4. Hysteresis thresholding (edge connection)
```

**Why switched:**
- `non_maximum_suppression()` requires pixel-by-pixel iteration
- Pure Python: ~30 seconds per image
- OpenCV: ~1 millisecond per image
- This is preprocessing, not the physics simulation

---

### 3. Otsu Threshold → `cv2.threshold(..., THRESH_OTSU)`

**Original Math Implementation:**
```python
def otsu_threshold():
    # Maximize between-class variance
    σ²ᵦ(t) = w₀(t) · w₁(t) · (μ₀(t) - μ₁(t))²
    
    For t in 0..255:
        Calculate class probabilities w₀, w₁
        Calculate class means μ₀, μ₁
        Find t that maximizes σ²ᵦ
```

**Why switched:** Same performance reason - image preprocessing, not core algorithm.

---

## SUMMARY TABLE

| Component | Implementation | Reason |
|-----------|---------------|--------|
| **RK4 Integration** | ✅ Pure Math | Core numerical method (project requirement) |
| **IVP Physics** | ✅ Pure Math | The differential equation we're solving |
| **Repulsive Forces** | ✅ Pure Math | Physics model |
| **Distance Matrix** | ✅ Pure Math | Simple, demonstrates math understanding |
| **Spatial Hashing** | ✅ Pure Math | Demonstrates data structure knowledge |
| **Greedy Assignment** | ✅ Pure Math | Demonstrates algorithm understanding |
| **Gaussian Blur** | ⚡ OpenCV | 1000x faster, just image preprocessing |
| **Canny Edges** | ⚡ OpenCV | 30000x faster, just image preprocessing |
| **Otsu Threshold** | ⚡ OpenCV | 100x faster, just image preprocessing |

---

## PERFORMANCE COMPARISON

| Version | Drones | GIF Frames | Runtime | Video Length |
|---------|--------|------------|---------|--------------|
| Pure Math (all) | 500 | 11 | ~5 min | 9 sec |
| Hybrid (current) | 1200 | 25 | ~2 min | 15.5 sec |
| Library (drone_show.py) | 1500 | 41 | ~30 sec | 14 sec |

---

## CONCLUSION

The **core numerical methods** (RK4, IVP model, collision avoidance) remain as pure mathematical implementations because:
1. They ARE the project deliverable
2. They demonstrate understanding of numerical methods
3. They're already efficient with numpy vectorization

The **image preprocessing** was switched to OpenCV because:
1. It's not the core algorithm being demonstrated
2. Pure Python image processing is impractically slow
3. The focus should be on physics simulation, not image I/O
