# Subtask 3: Dynamic Tracking (Animated GIF/Video)

## Complete Theoretical and Implementation Explanation

---

# 1. OVERVIEW

## What This Subtask Does

**Objective:** Drones continuously track a **moving target** — an animated shape that changes over time (e.g., a running tiger from a GIF).

**Key Difference from Subtasks 1 & 2:**
- Subtasks 1 & 2: Static targets (single shape, doesn't move)
- Subtask 3: Dynamic targets (shape changes every few frames)

**Visual Sequence:**
```
Time 0:                     Time 1:                     Time 2:
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│    ╱╲  ╱╲       │        │     ╱╲  ╱╲      │        │      ╱╲  ╱╲     │
│   ╱  ╲╱  ╲      │   →    │    ╱  ╲╱  ╲     │   →    │     ╱  ╲╱  ╲    │
│   │  ▓▓  │      │        │    │  ▓▓  │     │        │     │  ▓▓  │    │
│   ╲      ╱      │        │    ╲      ╱     │        │     ╲      ╱    │
│    ╲    ╱       │        │     ╲    ╱      │        │      ╲    ╱     │
└─────────────────┘        └─────────────────┘        └─────────────────┘
   Tiger frame 1              Tiger frame 2              Tiger frame 3
   Legs back                  Legs middle                Legs forward
```

The drones must **chase** the moving shape, never fully "arriving" because the target keeps changing.

---

# 2. MATHEMATICAL FORMULATION

## 2.1 Time-Varying Target

In Subtasks 1 & 2, targets were constant:
$$\vec{T}_i = \text{constant}$$

In Subtask 3, targets are **functions of time**:
$$\vec{T}_i(t) = \text{position from animation at time } t$$

## 2.2 Modified IVP

**Position:**
$$\frac{d\vec{x}_i}{dt} = \vec{v}_i \cdot \min\left(1, \frac{v_{max}}{|\vec{v}_i|}\right)$$

**Velocity:**
$$\frac{d\vec{v}_i}{dt} = \frac{1}{m}\left[k_p(\vec{T}_i(t) - \vec{x}_i) + \vec{F}_{rep,i} - k_d\vec{v}_i\right]$$

The target $\vec{T}_i(t)$ now **changes over time**, creating a "chasing" behavior.

## 2.3 Discrete Target Updates

In practice, we don't have continuous target motion. We have discrete animation frames:

```
GIF Frame 0 → targets T₀
GIF Frame 1 → targets T₁
GIF Frame 2 → targets T₂
...
```

Between GIF frames, we run several simulation steps with the same targets, then update to the next GIF frame.

---

# 3. TRACKING DYNAMICS

## 3.1 Steady-State Tracking Error

Unlike static targets where drones eventually reach their destination, dynamic targets create **perpetual tracking error**.

**Intuition:** If the target moves at velocity $\vec{v}_T$, the drone must also move at $\vec{v}_T$ to keep up. This requires a constant force, which means a constant displacement from the target.

**Tracking error (simplified analysis):**
$$\vec{e} = \vec{T} - \vec{x} = \frac{m \cdot \vec{v}_T}{k_p}$$

With our parameters and a target moving at 50 pixels/second:
$$e = \frac{1.0 \times 50}{25} = 2 \text{ pixels}$$

**Video Effect:** Drones lag slightly behind the moving shape. This creates a natural "chasing" appearance.

## 3.2 Phase Lag

The drone response lags behind target motion due to the system's time constant:

$$\tau = \frac{m}{k_d} = \frac{1}{12} \approx 0.083 \text{ seconds}$$

At 30 fps, this is about **2.5 frames** of lag.

**Video Effect:** When the tiger's leg moves, drones follow about 2-3 frames later.

---

# 4. CODE IMPLEMENTATION

## 4.1 Animation Loading

**Location:** Lines 527-580

### GIF Loading:
```python
def load_gif(path):
    """Load GIF frames as grayscale arrays."""
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(np.array(gif.convert('L')))
            gif.seek(gif.tell() + 1)
    except EOFError: 
        pass
    return frames
```

**How it works:**
1. Open GIF file with PIL
2. Loop through all frames (`gif.seek()` advances to next frame)
3. Convert each frame to grayscale ('L' mode)
4. Store as numpy arrays
5. Stop when `EOFError` (no more frames)

### Video Loading (MP4, AVI, etc.):
```python
def load_video(path):
    """Load video frames as grayscale arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    return frames
```

**How it works:**
1. Open video with OpenCV's `VideoCapture`
2. Read frames one by one
3. Convert BGR → grayscale
4. Store as numpy arrays
5. Stop when no more frames

### Unified Loader:
```python
def load_animation(path):
    """Automatically detect format and load."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.gif':
        return load_gif(path)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return load_video(path)
```

---

## 4.2 Frame Sampling

**Problem:** A typical GIF might have 50-100 frames. Processing all of them is slow and creates very long videos.

**Solution:** Sample a subset of frames.

```python
# Configuration
MAX_ANIMATION_FRAMES = 25  # Use at most 25 frames

# In main():
sample_rate = max(1, len(anim_frames) // MAX_ANIMATION_FRAMES)
sampled = anim_frames[::sample_rate]
```

**Example:**
- Original GIF: 75 frames
- sample_rate = 75 // 25 = 3
- Sampled: frames 0, 3, 6, 9, ... (25 frames)

---

## 4.3 The `track()` Method

**Location:** `Swarm.track()` (lines 662-673)

```python
def track(self, target_seq, steps=5):
    """Simulate dynamic tracking of moving targets."""
    result = []
    
    for t in target_seq:
        # Update targets to new animation frame
        self.tgt = greedy_assignment(self.pos, t)
        
        # Run simulation for a few steps
        for _ in range(steps):
            self.step()
            result.append(self.pos.copy())
    
    return result
```

### Algorithm:

```
For each animation frame:
    1. Extract target points from frame
    2. Reassign drones to new targets (greedy)
    3. Run physics simulation for STEPS_PER_FRAME steps
    4. Record drone positions
    
Repeat for all animation frames
```

### Parameters:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `STEPS_PER_FRAME` | 6 | Simulation steps between target updates |
| `MAX_ANIMATION_FRAMES` | 25 | Number of animation frames to use |

---

## 4.4 Phase 3 Execution in main()

**Location:** Lines 783-800

```python
# Phase 3: Dynamic Tracking
print("\n[Phase 3] Dynamic Tracking")
if PHASE3_ANIMATION:
    anim_path = os.path.join(data, PHASE3_ANIMATION)
    if os.path.exists(anim_path):
        anim_frames = load_animation(anim_path)
        
        # Sample frames
        sample_rate = max(1, len(anim_frames) // MAX_ANIMATION_FRAMES)
        sampled = anim_frames[::sample_rate]
        
        # Track the animation
        phase3_frames = swarm.track(
            [extract_points(f, n) for f in sampled], 
            steps=STEPS_PER_FRAME
        )
        frames.extend(phase3_frames)
```

**Pipeline:**
1. Load animation (GIF or video)
2. Sample frames (reduce to ~25)
3. For each sampled frame:
   - Extract edge points
   - Pass to `track()` method
4. Append all simulation frames to output

---

# 5. REASSIGNMENT AT EVERY FRAME

## 5.1 Why Reassign?

The shape changes between frames. Old assignments become suboptimal:

```
Frame N:          Frame N+1:
  ╱╲                 ╱╲
 ╱  ╲               ╱  ╲
╱    ╲    →        ╱    ╲
│    │            ╱      ╲
Legs together     Legs apart
```

Drones assigned to "leg together" position need new targets for "legs apart".

## 5.2 Computational Cost

Each frame requires:
1. `extract_points()`: Edge detection on new frame
2. `greedy_assignment()`: O(n²) assignment for n drones

For 25 frames with 1200 drones:
- 25 edge detections
- 25 × 1200² = 36 million distance comparisons

**This is why Subtask 3 takes longer to compute than Subtasks 1 & 2.**

## 5.3 Assignment Stability

Consecutive frames are usually similar, so most assignments stay the same:

```
Frame N: Drone 42 → Target at (350, 200)
Frame N+1: Drone 42 → Target at (352, 198)  (nearby)
```

The greedy algorithm naturally tends to pick nearby targets, creating smooth motion.

---

# 6. INTERPOLATION BETWEEN FRAMES

## 6.1 Why Multiple Steps?

If we only ran 1 simulation step per animation frame, drones would "teleport" between frames.

With `STEPS_PER_FRAME = 6`:
- Animation frame rate: ~5 fps (every 6 simulation frames)
- Simulation frame rate: 30 fps
- Drones have 6 steps to move toward new targets

## 6.2 Motion Smoothing

Between target updates, the physics naturally smooths motion:

```
Target jump:        Drone response:
Frame 10: T=(100,100)    x=(100,100)
Frame 11: T=(120,100)    x=(103,100)  (starts moving)
Frame 12: T=(120,100)    x=(108,100)  (accelerating)
Frame 13: T=(120,100)    x=(113,100)  (approaching)
Frame 14: T=(120,100)    x=(117,100)  (slowing)
Frame 15: T=(120,100)    x=(119,100)  (almost there)
Frame 16: T=(140,100)    x=(121,100)  (new target!)
```

The spring-damper system acts as a **low-pass filter**, smoothing out sudden target changes.

---

# 7. CONVERGENCE ANALYSIS

## 7.1 Never Fully Converged

Unlike Subtasks 1 & 2, drones in Subtask 3 **never fully reach their targets** because targets keep moving.

**Steady-state behavior:**
- Drones continuously move
- Average velocity ≈ target velocity
- Constant small tracking error

## 7.2 Tracking Quality Metrics

| Metric | Good | Poor |
|--------|------|------|
| Shape Recognition | Clear outline | Blurry blob |
| Lag | 2-3 frames | 10+ frames |
| Jitter | Smooth motion | Shaky/noisy |

**With our parameters:**
- Shape is recognizable
- Lag is 2-3 frames
- Motion is smooth

---

# 8. ANIMATION CONSIDERATIONS

## 8.1 Good Animation Properties

The tracking works best with animations that have:

| Property | Good | Bad |
|----------|------|-----|
| **Speed** | Moderate | Very fast |
| **Contrast** | High | Low (gray on gray) |
| **Edges** | Clear outlines | Blurry/soft |
| **Frame count** | 20-50 | <5 or >200 |
| **Resolution** | 200×200+ | Tiny (32×32) |

## 8.2 The Tiger GIF Example

The default animation is a running tiger:
- Clear silhouette (high contrast)
- Moderate speed
- Distinct frames (legs move)
- Good edge detection results

**Why it works well:** Each frame produces clear edge points, and the motion between frames is smooth enough for drones to track.

---

# 9. VIDEO OUTPUT FOR SUBTASK 3

## 9.1 Frame Count Calculation

```
Frames = (Number of animation frames) × (Steps per frame)
       = 25 × 6
       = 150 frames
       = 5 seconds at 30fps
```

**Note:** No "hold" phase in Subtask 3 — the animation loops or ends immediately.

## 9.2 Timeline

| Frame Range | Description |
|-------------|-------------|
| 0-6 | First animation frame, drones adjusting |
| 6-12 | Second animation frame |
| ... | Continuous tracking |
| 144-150 | Final animation frame |

## 9.3 Output Files

```
output/drone_show_math_phase3.mp4   - Only Subtask 3
output/drone_show_math_combined.mp4 - Full video (includes this)
```

---

# 10. COMPARISON: STATIC VS DYNAMIC TRACKING

| Aspect | Subtasks 1 & 2 (Static) | Subtask 3 (Dynamic) |
|--------|-------------------------|---------------------|
| **Target** | Constant | Time-varying |
| **Method** | `simulate()` | `track()` |
| **Assignments** | Once | Every frame |
| **Convergence** | Full (reaches target) | Partial (always chasing) |
| **Hold Phase** | Yes (40 frames) | No |
| **Velocity at End** | ~0 | Non-zero |
| **Computation** | Fast | Slower (many assignments) |

---

# 11. MATHEMATICAL INSIGHT: TRACKING VS REGULATION

## 11.1 Regulation (Subtasks 1 & 2)

The control objective is to reach and stay at a setpoint:
$$\lim_{t \to \infty} \vec{x}(t) = \vec{T}$$

This is called **regulation** or **setpoint control**.

## 11.2 Tracking (Subtask 3)

The control objective is to follow a moving reference:
$$\vec{x}(t) \approx \vec{T}(t) \text{ for all } t$$

This is called **tracking** or **trajectory following**.

## 11.3 Fundamental Limitation

For a 2nd-order system (position + velocity), tracking a constant-velocity target results in **steady-state error**:

$$e_{ss} = \frac{1}{K_p} \cdot \dot{T}$$

Where $\dot{T}$ is the target velocity.

**Physical meaning:** To move at constant velocity, you need constant force. Constant force requires constant displacement from equilibrium (the target).

---

# 12. CONFIGURATION OPTIONS

## 12.1 Animation File

```python
# GIF
PHASE3_ANIMATION = "my_animation.gif"

# Video
PHASE3_ANIMATION = "my_video.mp4"

# Disable Phase 3
PHASE3_ANIMATION = None
```

## 12.2 Performance Tuning

```python
# More animation frames (smoother but slower)
MAX_ANIMATION_FRAMES = 50

# Fewer simulation steps (faster but jumpier)
STEPS_PER_FRAME = 3
```

---

# 13. SUMMARY

## Subtask 3 Pipeline:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Load Animation  │ ──→ │ Sample Frames    │ ──→ │ Extract Points  │
│ (GIF/MP4)       │     │ (every Nth)      │     │ (per frame)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FOR EACH ANIMATION FRAME:                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Reassign    │ ──→ │ RK4 Steps   │ ──→ │ Record      │       │
│  │ (greedy)    │     │ (×6)        │     │ (positions) │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ↓
                    ┌─────────────────┐
                    │  Render Video   │
                    │  (MP4 output)   │
                    └─────────────────┘
```

## Key Differences from Subtasks 1 & 2:

1. **Targets change over time** — not static
2. **Reassignment every frame** — not once
3. **Continuous motion** — no settling/hold phase
4. **Tracking behavior** — always chasing, never arriving
5. **Supports GIF and video** — multiple animation formats
