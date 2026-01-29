# Subtask 2: Handwritten Name → Greeting Message

## Complete Theoretical and Implementation Explanation

---

# 1. OVERVIEW

## What This Subtask Does

**Objective:** Transform the drone swarm from your handwritten name (from Subtask 1) into a greeting message like "Happy New Year!".

**Visual Sequence:**
```
Frame 0 (from Sub1):        Frame 75:                   Frame 150:
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ ·····     ····  │        │    ·  ·    ··   │        │ ·  · ····  ···  │
│ ·   ·    ·   ·  │   →    │  ·   ·  · ·     │   →    │ ·  · ·  ·  ·  · │
│ ·····    ·····  │        │ ·· ·   ·   · ·  │        │ ···· ····  ···  │
│ ·  ·     ·   ·  │        │  ·  ·  · ·  ·   │        │ ·  · ·     ·    │
│ ·   ·    ·   ·  │        │   · ·    ·  · · │        │ ·  · ·     ·    │
└─────────────────┘        └─────────────────┘        └─────────────────┘
  "RAMAZ" (End Sub1)         Mid-Transition            "HNY" Complete
```

---

# 2. KEY DIFFERENCE FROM SUBTASK 1

## Starting Configuration

| Aspect | Subtask 1 | Subtask 2 |
|--------|-----------|-----------|
| **Initial State** | Uniform grid | Handwritten name shape |
| **Initial Velocity** | Zero (at rest) | Non-zero (from previous motion) |
| **Target** | Handwritten name | Greeting text |

**Critical Point:** In Subtask 2, drones are NOT starting from rest. They retain their velocities from the end of Subtask 1.

## Mathematical Implication

The IVP for Subtask 2 has different initial conditions:

$$\vec{x}_i(0) = \text{final position from Subtask 1}$$
$$\vec{v}_i(0) = \text{final velocity from Subtask 1} \approx \vec{0}$$

Since Subtask 1 includes a "hold" phase, velocities should be near zero, but the **positions** are now the name shape, not a grid.

---

# 3. MATHEMATICAL PROBLEM FORMULATION

## 3.1 State at Start of Subtask 2

After Subtask 1 completes:
- Drones are positioned on the handwritten name outline
- Drones have (nearly) zero velocity (settled)
- The swarm object retains its state

## 3.2 New Target Configuration

A new set of target points is extracted from the greeting image:

$$\vec{T}'_i = \text{points on "Happy New Year!" outline}$$

## 3.3 Re-Assignment Problem

The drone-to-target assignment must be recalculated because:
1. Drone positions have changed (now on name shape)
2. Target positions are different (greeting shape)
3. Optimal paths from new positions ≠ old paths

```
Before (Subtask 1):          After (Subtask 2):
Grid → Name                  Name → Greeting

  ···        ·····          ·····     Happy
  ···   →    ·   ·          ·   ·  →  New
  ···        ·····          ·····     Year!
```

---

# 4. THE PHYSICS ARE IDENTICAL

## Same IVP, Same Solver

The differential equations are **exactly the same** as Subtask 1:

**Position:**
$$\frac{d\vec{x}_i}{dt} = \vec{v}_i \cdot \min\left(1, \frac{v_{max}}{|\vec{v}_i|}\right)$$

**Velocity:**
$$\frac{d\vec{v}_i}{dt} = \frac{1}{m}\left[k_p(\vec{T}'_i - \vec{x}_i) + \vec{F}_{rep,i} - k_d\vec{v}_i\right]$$

The only difference is:
- $\vec{T}'_i$ is the **new** target (greeting, not name)
- Initial positions are the **name shape**, not grid

## Why This Works

The spring-mass-damper model doesn't care WHERE the target is. It just:
1. Computes force toward target
2. Applies damping
3. Avoids collisions

Same code, different inputs.

---

# 5. CODE IMPLEMENTATION

## 5.1 Phase 2 Execution in main()

**Location:** `main()` function (lines 763-781)

```python
# Phase 2: Name -> Greeting
print("\n[Phase 2] Greeting")
if PHASE2_IMAGE:
    gr = os.path.join(data, PHASE2_IMAGE)
    if os.path.exists(gr):
        print(f"  Loading image: {gr}")
        t2 = extract_points(gr, n)
    else:
        print(f"  Image not found: {gr}")
        print(f"  Using text fallback: '{PHASE2_TEXT_FALLBACK}'")
        t2 = extract_points(PHASE2_TEXT_FALLBACK, n, True)
else:
    print(f"  Using text: '{PHASE2_TEXT_FALLBACK}'")
    t2 = extract_points(PHASE2_TEXT_FALLBACK, n, True)

phase2_frames = swarm.simulate(t2, TRANSITION_FRAMES)
frames.extend(phase2_frames)
```

## 5.2 Swarm State Persistence

**Critical:** The `swarm` object is **not reset** between Subtask 1 and 2.

```python
swarm = Swarm(n)                      # Created once
phase1_frames = swarm.simulate(t1, ...)  # Subtask 1 - modifies swarm.pos
phase2_frames = swarm.simulate(t2, ...)  # Subtask 2 - starts from swarm.pos
```

After Subtask 1:
- `swarm.pos` = positions on handwritten name
- `swarm.vel` ≈ zero (settled during hold phase)
- `swarm.tgt` = old targets (will be overwritten)

Subtask 2 calls `simulate(t2, ...)`:
1. Computes new assignment: `swarm.tgt = greedy_assignment(swarm.pos, t2)`
2. Runs simulation from current positions
3. Drones fly from name → greeting

---

## 5.3 Target Extraction for Greeting

Same function as Subtask 1, but different input:

```python
# Text rendering for greeting
t2 = extract_points("Happy New Year!", n, is_text=True)
```

When `is_text=True`:

```python
def extract_points(source, n, is_text=False):
    if is_text:
        img = np.zeros((H, W), np.uint8)  # Black canvas
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5
        
        # Auto-scale to fit
        (tw, th), _ = cv2.getTextSize(source, font, scale, thick)
        if tw > W - 100: 
            scale *= (W - 100) / tw
        
        # Center text
        cv2.putText(img, source, ((W-tw)//2, (H+th)//2), font, scale, 255, thick)
        
        # Get white pixels as points
        pts = np.column_stack(np.where(img > 0))
```

**Result:** Points forming the rendered text "Happy New Year!"

---

# 6. REASSIGNMENT ANALYSIS

## 6.1 Why Reassignment Matters

The greedy assignment for Subtask 2 produces **different paths** than Subtask 1 because starting positions are different.

**Example:**

```
Subtask 1 (Grid → Name):     Subtask 2 (Name → Greeting):
Drone at (100, 200)          Drone at (400, 300)
  → Target (350, 250)          → Target (200, 150)
  Path: right + down           Path: left + up
```

## 6.2 Path Crossing Potential

When drones transition between very different shapes, paths may cross:

```
Name Shape:        Greeting Shape:      Paths:
    A                  H                  ╲╱
   ╱ ╲                ═╪═                  ╳
  ╱   ╲              │ │                  ╱╲
```

**Collision Avoidance Handles This:**
- When paths cross, drones get close to each other
- Repulsion forces push them apart
- They route around each other naturally

---

# 7. COLLISION DYNAMICS IN SUBTASK 2

## 7.1 Higher Collision Likelihood

In Subtask 1, drones start spread out on a grid → paths tend to be parallel.

In Subtask 2, drones start clustered on letter shapes → paths may converge/diverge dramatically.

**Example: "A" to "H" transition**
- Drones on left leg of "A" might go to left bar of "H"
- Drones on right leg of "A" might go to right bar of "H"
- Their paths cross in the middle

## 7.2 Repulsion Response

When drones approach within R_SAFE = 4 pixels:

$$\vec{F}_{rep} = \sum_{j: |\vec{x}_i - \vec{x}_j| < R_{safe}} k_{rep} \cdot \frac{\vec{x}_i - \vec{x}_j}{|\vec{x}_i - \vec{x}_j|^3}$$

**What happens:**
1. Drones slow down (damping)
2. Repulsion pushes them sideways
3. They curve around each other
4. Continue toward targets after clearing

**Video Effect:** Drones appear to "flow around" each other rather than collide.

---

# 8. CONVERGENCE ANALYSIS FOR SUBTASK 2

## 8.1 Is Convergence Different?

The mathematical analysis is **identical** to Subtask 1:

- Same damping ratio: $\zeta = 1.2$ (overdamped)
- Same time constant: $\tau \approx 0.37s$
- Same settling time: ~55 frames for 99%

## 8.2 Potential Difference: Distance to Target

If the greeting shape is very different from the name shape, average distance to target may be larger:

| Transition | Typical Distance | Frames Needed |
|------------|------------------|---------------|
| Grid → Name | ~200 pixels | ~50 frames |
| Name → Greeting | ~250 pixels | ~60 frames |
| Name → Very Different | ~400 pixels | ~80 frames |

With `TRANSITION_FRAMES = 150`, all cases have sufficient time.

---

# 9. CONFIGURATION OPTIONS FOR SUBTASK 2

## 9.1 Image vs Text

**Using image file:**
```python
PHASE2_IMAGE = "greeting.png"  # Your custom image
```

**Using rendered text:**
```python
PHASE2_IMAGE = None  # Will use text fallback
PHASE2_TEXT_FALLBACK = "Happy New Year!"
```

## 9.2 Custom Greeting

Edit in the configuration section:

```python
PHASE2_TEXT_FALLBACK = "HELLO 2025"  # Any text you want
```

The text will be auto-scaled to fit the canvas.

---

# 10. VIDEO OUTPUT FOR SUBTASK 2

## 10.1 Frame Count

| Phase | Frames | Duration at 30fps |
|-------|--------|-------------------|
| Transition | 150 | 5.0 seconds |
| Hold | 40 | 1.3 seconds |
| **Total** | **190** | **6.3 seconds** |

## 10.2 Timeline

| Frame Range | Description |
|-------------|-------------|
| 0-10 | Drones begin moving from name shape |
| 10-40 | Acceleration phase, letters dissolving |
| 40-90 | Peak velocity, crossing paths |
| 90-120 | Deceleration, greeting forming |
| 120-150 | Settling into final positions |
| 150-190 | Hold - greeting clearly visible |

## 10.3 Output Files

```
output/drone_show_math_phase2.mp4   - Only Subtask 2
output/drone_show_math_combined.mp4 - Full video (includes this)
```

---

# 11. COMPARISON: SUBTASK 1 VS SUBTASK 2

| Aspect | Subtask 1 | Subtask 2 |
|--------|-----------|-----------|
| **Start Shape** | Uniform grid | Handwritten name |
| **End Shape** | Handwritten name | Greeting text |
| **Initial Velocity** | Zero | Near-zero |
| **Path Complexity** | Low (parallel paths) | Medium (crossing possible) |
| **Collision Events** | Few | More |
| **Physics Model** | Same | Same |
| **Solver** | RK4 | RK4 |
| **Duration** | 6.3 seconds | 6.3 seconds |

---

# 12. SUMMARY

## Subtask 2 Pipeline:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Name Positions │ ──→ │ Extract Greeting │ ──→ │ Greedy Assign   │
│  (From Sub1)    │     │ (Text/Image)     │     │ (New Pairing)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ↓
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  RK4 Simulation │ ──→ │  Record Frames   │ ──→ │  Render Video   │
│  (Same physics) │     │  (positions)     │     │  (MP4 output)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Key Insight:

**Subtask 2 is mathematically identical to Subtask 1** — it's just a different transition:
- Different starting positions (name, not grid)
- Different target positions (greeting, not name)
- Same physics, same solver, same parameters

The only "new" challenge is handling more path crossings, which the repulsion forces naturally resolve.
