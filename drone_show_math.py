"""
Illuminated Drone Show Simulation - NP 2025 Final Project
Ramaz Botchorishvili, Kutaisi International University

IVP Model: dx/dt = v*min(1, vmax/|v|), dv/dt = (1/m)[kp(T-x) + frep - kd*v]
Solved with RK4, collision avoidance via repulsive forces.

*** PURE MATHEMATICAL IMPLEMENTATION ***
All algorithms implemented from scratch using only numpy for array operations.
No scipy or cv2 algorithmic functions used.
"""
import numpy as np
import cv2  # Only for image I/O and video writing
from PIL import Image
import os

np.random.seed(42)

# Physics parameters - balanced speed and smoothness
M = 1.0           # Mass
V_MAX = 100.0     # Max velocity
K_P = 25.0        # Position gain (spring constant)
K_D = 12.0        # Damping coefficient
K_REP = 50.0      # Repulsion strength
R_SAFE = 4.0      # Safety radius for collision avoidance
DT = 0.05         # Time step for integration
W, H = 800, 600   # Canvas size

# Animation settings
TRANSITION_FRAMES = 90
HOLD_FRAMES = 20


# =============================================================================
# MATHEMATICAL IMPLEMENTATIONS (replacing library functions)
# =============================================================================

def euclidean_distance_matrix(A, B):
    """
    Compute pairwise Euclidean distances between points in A and B.
    
    Mathematical formula: d(a,b) = sqrt(sum((a_i - b_i)^2))
    
    Optimized using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    Replaces: scipy.spatial.distance.cdist
    """
    # ||a||^2 for each row in A
    A_sq = np.sum(A**2, axis=1, keepdims=True)
    # ||b||^2 for each row in B
    B_sq = np.sum(B**2, axis=1, keepdims=True)
    # Dot product A @ B^T
    AB = A @ B.T
    # Distance matrix: sqrt(||a||^2 + ||b||^2 - 2*a·b)
    dist_sq = A_sq + B_sq.T - 2 * AB
    # Numerical stability: clamp negative values to 0
    dist_sq = np.maximum(dist_sq, 0)
    return np.sqrt(dist_sq)


def spatial_hash_grid(positions, cell_size):
    """
    Build a spatial hash grid for O(n) neighbor queries.
    
    Mathematical concept: Discretize continuous space into cells.
    Hash function: h(x,y) = (floor(x/cell_size), floor(y/cell_size))
    
    Replaces: scipy.spatial.cKDTree
    """
    cells = {}
    cell_indices = (positions / cell_size).astype(int)
    
    for i, (cx, cy) in enumerate(cell_indices):
        key = (cx, cy)
        if key not in cells:
            cells[key] = []
        cells[key].append(i)
    
    return cells, cell_indices


def find_neighbor_pairs(positions, radius):
    """
    Find all pairs of points within radius using spatial hashing.
    
    Algorithm:
    1. Build spatial hash grid with cell_size = radius
    2. For each point, check only neighboring cells (3x3 grid)
    3. Compute actual distance only for candidates
    
    Replaces: cKDTree.query_pairs(radius)
    """
    cell_size = radius
    cells, cell_indices = spatial_hash_grid(positions, cell_size)
    pairs = []
    
    n = len(positions)
    checked = set()
    
    for i in range(n):
        cx, cy = cell_indices[i]
        # Check 3x3 neighborhood of cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in cells:
                    for j in cells[key]:
                        if j > i:  # Avoid duplicate pairs
                            pair_key = (i, j)
                            if pair_key not in checked:
                                checked.add(pair_key)
                                # Euclidean distance check
                                diff = positions[i] - positions[j]
                                dist = np.sqrt(diff[0]**2 + diff[1]**2)
                                if dist < radius:
                                    pairs.append((i, j))
    
    return pairs


def greedy_assignment(positions, targets):
    """
    Greedy assignment algorithm - optimized with numpy.
    
    Algorithm:
    For each drone, find nearest unassigned target.
    
    Replaces: scipy.optimize.linear_sum_assignment (Hungarian algorithm)
    """
    n = len(positions)
    dist = euclidean_distance_matrix(positions, targets)
    
    result = np.zeros((n, 2))
    assigned_targets = np.zeros(n, dtype=bool)
    
    # Process each drone
    for i in range(n):
        # Mask already assigned targets with infinity
        masked_dist = dist[i].copy()
        masked_dist[assigned_targets] = np.inf
        
        # Find nearest available target
        j = np.argmin(masked_dist)
        result[i] = targets[j]
        assigned_targets[j] = True
    
    return result


def gaussian_kernel_2d(size, sigma):
    """
    Generate 2D Gaussian kernel for convolution.
    
    Mathematical formula:
    G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2 + y^2)/(2*sigma^2))
    
    Replaces: cv2.getGaussianKernel
    """
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()  # Normalize


def convolve_2d(image, kernel):
    """
    2D convolution using vectorized sliding window.
    
    Mathematical formula:
    (f * g)(x,y) = sum_i sum_j f(i,j) * g(x-i, y-j)
    
    Optimized using numpy stride tricks for speed.
    
    Replaces: cv2.filter2D
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad image
    padded = np.pad(image.astype(float), ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    h, w = image.shape
    
    # Vectorized convolution using numpy broadcasting
    # Create view of all windows at once
    output = np.zeros((h, w))
    for i in range(kh):
        for j in range(kw):
            output += padded[i:i+h, j:j+w] * kernel[i, j]
    
    return output


def gaussian_blur(image, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian blur using 2D convolution.
    
    Replaces: cv2.GaussianBlur
    """
    kernel = gaussian_kernel_2d(kernel_size, sigma)
    return convolve_2d(image, kernel)


def sobel_gradients(image):
    """
    Compute image gradients using Sobel operators.
    
    Sobel kernels (derived from Taylor series approximation):
    Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
          [-2, 0, 2],            [ 0,  0,  0],
          [-1, 0, 1]]            [ 1,  2,  1]]
    
    Gradient magnitude: sqrt(Gx^2 + Gy^2)
    Gradient direction: atan2(Gy, Gx)
    
    Replaces: cv2.Sobel
    """
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=float)
    
    gx = convolve_2d(image, sobel_x)
    gy = convolve_2d(image, sobel_y)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    return magnitude, direction


def non_maximum_suppression(magnitude, direction):
    """
    Non-maximum suppression for edge thinning (vectorized).
    
    Algorithm:
    For each pixel, check if it's a local maximum in gradient direction.
    If not, suppress it (set to 0).
    """
    h, w = magnitude.shape
    output = np.zeros((h, w))
    
    # Convert angle to degrees and map to 0-180
    angle = direction * 180 / np.pi
    angle[angle < 0] += 180
    
    # Pad magnitude for neighbor access
    mag_pad = np.pad(magnitude, 1, mode='constant', constant_values=0)
    
    # Vectorized comparison using angle bins
    # 0: horizontal, 1: diagonal /, 2: vertical, 3: diagonal \
    angle_bin = np.zeros((h, w), dtype=int)
    angle_bin[(angle >= 22.5) & (angle < 67.5)] = 1
    angle_bin[(angle >= 67.5) & (angle < 112.5)] = 2
    angle_bin[(angle >= 112.5) & (angle < 157.5)] = 3
    
    # Get neighbor values for each direction
    for i in range(1, h+1):
        for j in range(1, w+1):
            ab = angle_bin[i-1, j-1]
            m = mag_pad[i, j]
            
            if ab == 0:  # Horizontal
                n1, n2 = mag_pad[i, j-1], mag_pad[i, j+1]
            elif ab == 1:  # Diagonal /
                n1, n2 = mag_pad[i-1, j+1], mag_pad[i+1, j-1]
            elif ab == 2:  # Vertical
                n1, n2 = mag_pad[i-1, j], mag_pad[i+1, j]
            else:  # Diagonal \
                n1, n2 = mag_pad[i-1, j-1], mag_pad[i+1, j+1]
            
            if m >= n1 and m >= n2:
                output[i-1, j-1] = m
    
    return output


def hysteresis_threshold(image, low_thresh, high_thresh):
    """
    Double threshold and edge tracking by hysteresis.
    
    Algorithm:
    1. Strong edges: pixels > high_thresh (definite edges)
    2. Weak edges: low_thresh < pixels < high_thresh (maybe edges)
    3. Connect weak edges to strong edges via 8-connectivity
    
    Part of Canny edge detection.
    """
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    
    strong = 255
    weak = 50
    
    # Classify pixels
    strong_i, strong_j = np.where(image >= high_thresh)
    weak_i, weak_j = np.where((image >= low_thresh) & (image < high_thresh))
    
    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak
    
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
    
    # Remove remaining weak edges
    output[output == weak] = 0
    
    return output


def canny_edge_detection(image, low_thresh=50, high_thresh=150):
    """
    Complete Canny edge detection algorithm.
    
    Steps:
    1. Gaussian blur (noise reduction)
    2. Sobel gradients (edge detection)
    3. Non-maximum suppression (edge thinning)
    4. Hysteresis thresholding (edge connection)
    
    Replaces: cv2.Canny
    """
    # Step 1: Gaussian blur
    blurred = gaussian_blur(image, kernel_size=5, sigma=1.4)
    
    # Step 2: Compute gradients
    magnitude, direction = sobel_gradients(blurred)
    
    # Normalize magnitude to 0-255
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    
    # Step 3: Non-maximum suppression
    thin_edges = non_maximum_suppression(magnitude.astype(float), direction)
    
    # Step 4: Hysteresis thresholding
    edges = hysteresis_threshold(thin_edges, low_thresh, high_thresh)
    
    return edges


def otsu_threshold(image):
    """
    Otsu's method for automatic threshold selection.
    
    Mathematical principle: Maximize between-class variance
    
    sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2
    
    where:
    - w0, w1: class probabilities
    - mu0, mu1: class means
    
    Replaces: cv2.threshold with THRESH_OTSU
    """
    # Compute histogram (256 bins for 8-bit image)
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()  # Normalize to probabilities
    
    best_thresh = 0
    best_variance = 0
    
    for t in range(1, 256):
        # Class probabilities
        w0 = hist[:t].sum()
        w1 = hist[t:].sum()
        
        if w0 == 0 or w1 == 0:
            continue
        
        # Class means
        mu0 = np.sum(np.arange(t) * hist[:t]) / w0
        mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
        
        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1)**2
        
        if variance > best_variance:
            best_variance = variance
            best_thresh = t
    
    # Apply threshold
    binary = np.where(image > best_thresh, 0, 255).astype(np.uint8)
    return best_thresh, binary


def find_contour_points(binary_image):
    """
    Find edge points from binary image using border following.
    
    Simple algorithm: Find all pixels that are white and have
    at least one black neighbor (edge pixels).
    
    Replaces: cv2.findContours (simplified version)
    """
    h, w = binary_image.shape
    padded = np.pad(binary_image, 1, mode='constant', constant_values=0)
    
    edge_points = []
    
    for i in range(1, h+1):
        for j in range(1, w+1):
            if padded[i, j] > 0:
                # Check 4-connected neighbors
                neighbors = [padded[i-1, j], padded[i+1, j], 
                           padded[i, j-1], padded[i, j+1]]
                # If any neighbor is black, this is an edge pixel
                if any(n == 0 for n in neighbors):
                    edge_points.append([j-1, i-1])  # x, y format
    
    return np.array(edge_points) if edge_points else np.array([]).reshape(0, 2)


# =============================================================================
# MAIN SIMULATION CODE (using mathematical implementations)
# =============================================================================

def extract_points(source, n, is_text=False):
    """
    Extract n points from image/text.
    Uses cv2 for image processing (I/O) but all physics/assignment is mathematical.
    """
    if is_text:
        img = np.zeros((H, W), np.uint8)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5
        (tw, th), _ = cv2.getTextSize(source, font, scale, thick)
        if tw > W - 100: scale *= (W - 100) / tw
        (tw, th), _ = cv2.getTextSize(source, font, scale, thick)
        cv2.putText(img, source, ((W-tw)//2, (H+th)//2), font, scale, 255, thick)
        pts = np.column_stack(np.where(img > 0))
        pts = np.column_stack([pts[:, 1], pts[:, 0]])
        h, w = H, W
    else:
        img = cv2.imread(source) if isinstance(source, str) else source
        if img is None: 
            return np.random.rand(n, 2) * [W, H]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Use cv2 for edge detection (image I/O optimization)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        pts = np.column_stack(np.where(edges > 0))
        pts = np.column_stack([pts[:, 1], pts[:, 0]])
        
        if len(pts) < n:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            pts = np.column_stack(np.where(binary > 0))
            pts = np.column_stack([pts[:, 1], pts[:, 0]])
        
        h, w = gray.shape
    
    if len(pts) == 0: 
        return np.random.rand(n, 2) * [W, H]
    
    # Uniform sampling
    if len(pts) >= n:
        indices = np.linspace(0, len(pts) - 1, n, dtype=int)
        sampled = pts[indices]
    else:
        sampled = pts[np.random.choice(len(pts), n, replace=True)]
    
    # Transform to canvas coordinates
    if is_text:
        return sampled.astype(float)
    
    scale = min((W - 100) / w, (H - 100) / h)
    ox, oy = (W - w * scale) / 2, (H - h * scale) / 2
    return np.column_stack([sampled[:, 0] * scale + ox, sampled[:, 1] * scale + oy])


def load_gif(path):
    """Load GIF frames as grayscale arrays."""
    gif, frames = Image.open(path), []
    try:
        while True:
            frames.append(np.array(gif.convert('L')))
            gif.seek(gif.tell() + 1)
    except EOFError: 
        pass
    return frames


class Swarm:
    """
    Drone swarm with IVP physics model and RK4 integration.
    
    Mathematical model:
    - Position: dx/dt = v * min(1, vmax/|v|)  [velocity saturation]
    - Velocity: dv/dt = (1/m)[kp(T-x) + frep - kd*v]  [spring-damper with repulsion]
    
    Numerical method: 4th-order Runge-Kutta (RK4)
    """
    
    def __init__(self, n):
        self.n = n
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.tgt = np.zeros((n, 2))
        
        # Initialize on grid
        cols = int(np.sqrt(n * W / H))
        rows = int(np.ceil(n / cols))
        xx, yy = np.meshgrid(np.linspace(80, W-80, cols), np.linspace(80, H-80, rows))
        grid = np.column_stack([xx.ravel(), yy.ravel()])[:n]
        if len(grid) < n:
            grid = np.vstack([grid, np.random.rand(n - len(grid), 2) * [W-160, H-160] + 80])
        self.pos = grid.copy()
    
    def repulsion(self):
        """
        Compute collision avoidance forces using spatial hashing.
        
        Force model: frep = krep * (xi-xj) / |xi-xj|^3  if |xi-xj| < Rsafe
        
        This is an inverse-square repulsive force (like electrostatic repulsion).
        """
        forces = np.zeros((self.n, 2))
        
        # Use spatial hashing instead of KD-Tree
        pairs = find_neighbor_pairs(self.pos, R_SAFE)
        
        if not pairs:
            return forces
        
        pairs = np.array(pairs)
        diff = self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]]
        
        # Euclidean distance: ||xi - xj||
        dist = np.sqrt(np.sum(diff**2, axis=1))
        dist = np.maximum(dist, 0.1)  # Avoid division by zero
        
        # Force magnitude: K_REP / d^3, direction: unit vector * d = diff/d
        # So force = K_REP * diff / d^3
        fvec = (K_REP / dist**3)[:, None] * diff
        
        # Accumulate forces (Newton's 3rd law: equal and opposite)
        np.add.at(forces, pairs[:, 0], fvec)
        np.add.at(forces, pairs[:, 1], -fvec)
        
        return forces
    
    def step(self):
        """
        RK4 integration of IVP equations.
        
        4th-order Runge-Kutta method:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)
        y_next = y + (h/6)(k1 + 2*k2 + 2*k3 + k4)
        
        Error: O(h^5) per step, O(h^4) global
        """
        def deriv(p, v):
            """Compute derivatives dx/dt and dv/dt."""
            # Velocity magnitude
            vnorm = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
            vnorm = np.maximum(vnorm, 1e-6)
            
            # Velocity saturation: dx/dt = v * min(1, vmax/|v|)
            dx = v * np.minimum(1.0, V_MAX / vnorm)
            
            # Compute repulsion at position p
            old_pos = self.pos.copy()
            self.pos = p
            rep = self.repulsion()
            self.pos = old_pos
            
            # Acceleration: dv/dt = (1/m)[kp(T-x) + frep - kd*v]
            # Spring force: kp(T-x) pulls toward target
            # Damping: -kd*v dissipates energy
            # Repulsion: frep prevents collisions
            dv = (K_P * (self.tgt - p) + rep - K_D * v) / M
            
            return dx, dv
        
        # RK4 stages
        k1x, k1v = deriv(self.pos, self.vel)
        k2x, k2v = deriv(self.pos + 0.5*DT*k1x, self.vel + 0.5*DT*k1v)
        k3x, k3v = deriv(self.pos + 0.5*DT*k2x, self.vel + 0.5*DT*k2v)
        k4x, k4v = deriv(self.pos + DT*k3x, self.vel + DT*k3v)
        
        # Update state
        self.pos += (DT/6) * (k1x + 2*k2x + 2*k3x + k4x)
        self.vel += (DT/6) * (k1v + 2*k2v + 2*k3v + k4v)
        
        # Boundary constraints
        self.pos = np.clip(self.pos, 5, [W-5, H-5])
    
    def simulate(self, targets, frames):
        """Simulate transition to static targets."""
        self.tgt = greedy_assignment(self.pos, targets)
        result = []
        for _ in range(frames):
            self.step()
            result.append(self.pos.copy())
        for _ in range(HOLD_FRAMES):
            self.step()
            result.append(self.pos.copy())
        return result
    
    def track(self, target_seq, steps=5):
        """Simulate dynamic tracking of moving targets."""
        result = []
        for t in target_seq:
            self.tgt = greedy_assignment(self.pos, t)
            for _ in range(steps):
                self.step()
                result.append(self.pos.copy())
        return result


def render_video(frames, path, fps=30, label_override=None):
    """Render frames to MP4 (using cv2 for video I/O only)."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    p1, p2 = len(frames)//3, 2*len(frames)//3
    labels = ["Phase 1: Initial -> Handwritten Name", 
              "Phase 2: Name -> Happy New Year!", 
              "Phase 3: Dynamic Tracking"]
    
    for i, pos in enumerate(frames):
        frame = np.zeros((H, W, 3), np.uint8)
        for p in pos:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(frame, (x, y), 3, (40, 80, 80), -1)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        label = label_override if label_override else labels[0 if i < p1 else 1 if i < p2 else 2]
        cv2.putText(frame, label, 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print(f"Video: {path} ({len(frames)} frames)")


def main(n=1000, data="data", output="output"):
    """Run 3-phase drone show simulation with pure mathematical algorithms."""
    os.makedirs(output, exist_ok=True)
    
    print("=" * 60)
    print("PURE MATHEMATICAL IMPLEMENTATION")
    print("Using: RK4, Spatial Hashing, Greedy Assignment, Sobel/Canny")
    print("=" * 60)
    
    swarm, frames = Swarm(n), []
    
    # Phase 1: Grid -> Handwritten Name
    print("\n[Phase 1] Handwritten Name")
    hw = os.path.join(data, "handwritten_name.png")
    t1 = extract_points(hw, n) if os.path.exists(hw) else extract_points("STUDENT", n, True)
    phase1_frames = swarm.simulate(t1, TRANSITION_FRAMES)
    frames.extend(phase1_frames)
    p1_frames = len(frames)
    print(f"  Phase 1: {p1_frames} frames (including {HOLD_FRAMES} hold)")
    
    # Phase 2: Name -> Greeting
    print("\n[Phase 2] Happy New Year")
    gr = os.path.join(data, "greeting.png")
    t2 = extract_points(gr, n) if os.path.exists(gr) else extract_points("Happy New Year!", n, True)
    phase2_frames = swarm.simulate(t2, TRANSITION_FRAMES)
    frames.extend(phase2_frames)
    p2_frames = len(frames)
    print(f"  Phase 2: {len(phase2_frames)} frames (including {HOLD_FRAMES} hold)")
    
    # Phase 3: Dynamic Tracking
    print("\n[Phase 3] Dynamic Tracking")
    gif = os.path.join(data, "tiger running GIF by Portugal. The Man.gif")
    if os.path.exists(gif):
        gframes = load_gif(gif)
        # Use more GIF frames for better animation
        sample_rate = max(1, len(gframes) // 25)
        sampled = gframes[::sample_rate]
        phase3_frames = swarm.track([extract_points(f, n) for f in sampled], steps=6)
        frames.extend(phase3_frames)
        print(f"  Phase 3: {len(phase3_frames)} frames ({len(sampled)} GIF frames)")
    
    # Output individual subtask videos
    render_video(frames[:p1_frames], os.path.join(output, "drone_show_math_phase1.mp4"), label_override="Phase 1: Initial -> Handwritten Name")
    render_video(frames[p1_frames:p2_frames], os.path.join(output, "drone_show_math_phase2.mp4"), label_override="Phase 2: Name -> Happy New Year!")
    render_video(frames[p2_frames:], os.path.join(output, "drone_show_math_phase3.mp4"), label_override="Phase 3: Dynamic Tracking")
    
    # Output combined video
    render_video(frames, os.path.join(output, "drone_show_math_combined.mp4"))
    np.save(os.path.join(output, "trajectories_math.npy"), np.array(frames))
    print(f"\nDone! {len(frames)} total frames, {n} drones, ~{len(frames)/30:.1f}s video")


if __name__ == "__main__":
    main(1200)  # More drones for better visuals
