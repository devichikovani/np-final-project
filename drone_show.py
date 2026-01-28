"""
Illuminated Drone Show Simulation - NP 2025 Final Project
Ramaz Botchorishvili, Kutaisi International University

IVP Model: dx/dt = v*min(1, vmax/|v|), dv/dt = (1/m)[kp(T-x) + frep - kd*v]
Solved with RK4, collision avoidance via repulsive forces.
"""
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os

np.random.seed(42)

# Physics parameters - balanced speed and smoothness
M = 1.0           # Mass
V_MAX = 100.0     # Max velocity (faster)
K_P = 25.0        # Position gain (responsive)
K_D = 12.0        # Damping (smooth)
K_REP = 50.0      # Repulsion
R_SAFE = 4.0      # Safety radius
DT = 0.05         # Time step
W, H = 800, 600   # Canvas size

# Animation settings
TRANSITION_FRAMES = 90    # Quick transitions
HOLD_FRAMES = 20          # Brief hold


def extract_points(source, n, is_text=False):
    """Extract n points from image/text using precise contour detection."""
    if is_text:
        img = np.zeros((H, W), np.uint8)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5
        (tw, th), _ = cv2.getTextSize(source, font, scale, thick)
        if tw > W - 100: scale *= (W - 100) / tw
        (tw, th), _ = cv2.getTextSize(source, font, scale, thick)
        cv2.putText(img, source, ((W-tw)//2, (H+th)//2), font, scale, 255, thick)
        # Find contours for precise edge following
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pts = np.vstack([c.reshape(-1, 2) for c in contours]) if contours else np.column_stack(np.where(img > 0))
        if len(pts) < n:
            pts = np.column_stack(np.where(img > 0))
            pts = np.column_stack([pts[:, 1], pts[:, 0]])
        h, w = H, W
        is_contour = True
    else:
        img = cv2.imread(source) if isinstance(source, str) else source
        if img is None: return np.random.rand(n, 2) * [W, H]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Enhanced edge detection with multiple methods
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding for better edge detection
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Find contours for precise edge following
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            # Get all contour points
            pts = np.vstack([c.reshape(-1, 2) for c in contours])
            is_contour = True
        else:
            pts = np.column_stack(np.where(edges > 0))
            is_contour = False
        
        # Fallback to binary threshold if not enough points
        if len(pts) < n:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if contours:
                pts = np.vstack([c.reshape(-1, 2) for c in contours])
                is_contour = True
            else:
                pts = np.column_stack(np.where(binary > 0))
                is_contour = False
        
        h, w = gray.shape
    
    if len(pts) == 0: return np.random.rand(n, 2) * [W, H]
    
    # Uniform sampling along contours for even distribution
    if len(pts) >= n:
        # Sample uniformly along the contour
        indices = np.linspace(0, len(pts) - 1, n, dtype=int)
        sampled = pts[indices]
    else:
        # Need to repeat points
        sampled = pts[np.random.choice(len(pts), n, replace=True)]
    
    # Transform to canvas coordinates
    if is_text:
        return sampled.astype(float)
    
    pad, scale = 50, min((W - 100) / w, (H - 100) / h)
    ox, oy = (W - w * scale) / 2, (H - h * scale) / 2
    
    if is_contour:
        # Contour points are (x, y) format
        return np.column_stack([sampled[:, 0] * scale + ox, sampled[:, 1] * scale + oy])
    else:
        # np.where gives (row, col) = (y, x)
        return np.column_stack([sampled[:, 1] * scale + ox, sampled[:, 0] * scale + oy])


def load_gif(path):
    """Load GIF frames as grayscale arrays."""
    gif, frames = Image.open(path), []
    try:
        while True:
            frames.append(np.array(gif.convert('L')))
            gif.seek(gif.tell() + 1)
    except EOFError: pass
    return frames


def load_video(path):
    """Load video frames as grayscale arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frames.append(gray)
    cap.release()
    return frames


def assign_targets(pos, targets):
    """Optimal assignment using Hungarian algorithm - minimizes total distance."""
    n = len(pos)
    
    # For large n, use chunked assignment for speed
    if n > 500:
        # Divide into chunks and assign optimally within chunks
        chunk_size = 250
        result = np.zeros_like(targets)
        assigned = np.zeros(n, dtype=bool)
        
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_pos = pos[start:end]
            
            # Find unassigned targets
            avail_idx = np.where(~assigned)[0]
            avail_targets = targets[avail_idx]
            
            # Compute cost matrix for this chunk
            cost = cdist(chunk_pos, avail_targets)
            row_ind, col_ind = linear_sum_assignment(cost)
            
            for r, c in zip(row_ind, col_ind):
                result[start + r] = avail_targets[c]
                assigned[avail_idx[c]] = True
        
        return result
    else:
        # Full optimal assignment for smaller swarms
        cost = cdist(pos, targets)
        row_ind, col_ind = linear_sum_assignment(cost)
        return targets[col_ind]


class Swarm:
    """Drone swarm with IVP physics model and RK4 integration."""
    
    def __init__(self, n):
        self.n, self.pos, self.vel, self.tgt = n, np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
        # Initialize grid
        cols = int(np.sqrt(n * W / H))
        rows = int(np.ceil(n / cols))
        xx, yy = np.meshgrid(np.linspace(80, W-80, cols), np.linspace(80, H-80, rows))
        grid = np.column_stack([xx.ravel(), yy.ravel()])[:n]
        if len(grid) < n:
            grid = np.vstack([grid, np.random.rand(n - len(grid), 2) * [W-160, H-160] + 80])
        self.pos = grid.copy()
    
    def repulsion(self):
        """Compute collision avoidance forces: frep = krep * (xi-xj) / |xi-xj|^3 if |xi-xj| < Rsafe."""
        forces, pairs = np.zeros((self.n, 2)), list(cKDTree(self.pos).query_pairs(R_SAFE))
        if not pairs: return forces
        pairs = np.array(pairs)
        diff = self.pos[pairs[:, 0]] - self.pos[pairs[:, 1]]
        dist = np.maximum(np.linalg.norm(diff, axis=1), 0.1)
        fvec = (K_REP / dist**3)[:, None] * (diff / dist[:, None])
        np.add.at(forces, pairs[:, 0], fvec)
        np.add.at(forces, pairs[:, 1], -fvec)
        return forces
    
    def step(self):
        """RK4 integration of IVP equations."""
        def deriv(p, v):
            vnorm = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
            dx = v * np.minimum(1.0, V_MAX / vnorm)  # Velocity saturation
            old, self.pos = self.pos.copy(), p
            rep = self.repulsion()
            self.pos = old
            dv = (K_P * (self.tgt - p) + rep - K_D * v) / M  # IVP equation
            return dx, dv
        
        k1x, k1v = deriv(self.pos, self.vel)
        k2x, k2v = deriv(self.pos + 0.5*DT*k1x, self.vel + 0.5*DT*k1v)
        k3x, k3v = deriv(self.pos + 0.5*DT*k2x, self.vel + 0.5*DT*k2v)
        k4x, k4v = deriv(self.pos + DT*k3x, self.vel + DT*k3v)
        self.pos += (DT/6) * (k1x + 2*k2x + 2*k3x + k4x)
        self.vel += (DT/6) * (k1v + 2*k2v + 2*k3v + k4v)
        self.pos = np.clip(self.pos, 5, [W-5, H-5])
    
    def simulate(self, targets, frames):
        """Simulate transition to static targets with settling."""
        self.tgt = assign_targets(self.pos, targets)
        result = []
        for _ in range(frames):
            self.step()
            result.append(self.pos.copy())
        # Add hold frames once settled
        for _ in range(HOLD_FRAMES):
            self.step()
            result.append(self.pos.copy())
        return result
    
    def track(self, target_seq, steps=5):
        """Simulate dynamic tracking of moving targets."""
        result = []
        for t in target_seq:
            self.tgt = assign_targets(self.pos, t)
            for _ in range(steps):
                self.step()
                result.append(self.pos.copy())
        return result


def render_video(frames, path, fps=30, label_override=None):
    """Render frames to MP4 with illuminated drones."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    p1, p2 = len(frames)//3, 2*len(frames)//3
    labels = ["Phase 1: Initial -> Handwritten Name", "Phase 2: Name -> Happy New Year!", "Phase 3: Dynamic Tracking"]
    
    for i, pos in enumerate(frames):
        frame = np.zeros((H, W, 3), np.uint8)
        for p in pos:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(frame, (x, y), 3, (40, 80, 80), -1)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        label = label_override if label_override else labels[0 if i < p1 else 1 if i < p2 else 2]
        cv2.putText(frame, label, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print(f"Video: {path} ({len(frames)} frames)")


def main(n=1000, data="data", output="output"):
    """Run 3-phase drone show simulation."""
    os.makedirs(output, exist_ok=True)
    swarm, frames = Swarm(n), []
    
    # Phase 1: Grid -> Handwritten Name
    print("[Phase 1] Handwritten Name")
    hw = os.path.join(data, "handwritten_name.png")
    t1 = extract_points(hw, n) if os.path.exists(hw) else extract_points("STUDENT", n, True)
    frames.extend(swarm.simulate(t1, TRANSITION_FRAMES))
    print(f"  Phase 1: {len(frames)} frames (including {HOLD_FRAMES} hold)")
    
    # Phase 2: Name -> Greeting
    print("[Phase 2] Happy New Year")
    gr = os.path.join(data, "greeting.png")
    t2 = extract_points(gr, n) if os.path.exists(gr) else extract_points("Happy New Year!", n, True)
    p1_frames = len(frames)
    frames.extend(swarm.simulate(t2, TRANSITION_FRAMES))
    print(f"  Phase 2: {len(frames) - p1_frames} frames (including {HOLD_FRAMES} hold)")
    
    # Phase 3: Dynamic Tracking
    print("[Phase 3] Dynamic Tracking")
    gif = os.path.join(data, "tiger running GIF by Portugal. The Man.gif")
    if os.path.exists(gif):
        gframes = load_video(gif)
        # Use ALL GIF frames with smooth interpolation
        sampled = gframes
        p2_frames = len(frames)
        # 5 physics steps per GIF frame - fast but smooth
        frames.extend(swarm.track([extract_points(f, n) for f in sampled], steps=5))
        print(f"  Phase 3: {len(frames) - p2_frames} frames ({len(sampled)} GIF frames)")
    
    # Output individual subtask videos
    render_video(frames[:p1_frames], os.path.join(output, "drone_show_phase1.mp4"), label_override="Phase 1: Initial -> Handwritten Name")
    render_video(frames[p1_frames:p2_frames], os.path.join(output, "drone_show_phase2.mp4"), label_override="Phase 2: Name -> Happy New Year!")
    render_video(frames[p2_frames:], os.path.join(output, "drone_show_phase3.mp4"), label_override="Phase 3: Dynamic Tracking")
    
    # Output combined video
    render_video(frames, os.path.join(output, "drone_show_combined.mp4"))
    np.save(os.path.join(output, "trajectories.npy"), np.array(frames))
    print(f"Done! {len(frames)} total frames, {n} drones, ~{len(frames)/30:.1f}s video")


if __name__ == "__main__":
    main(1500)
