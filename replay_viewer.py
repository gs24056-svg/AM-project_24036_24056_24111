import csv
import math
import sys
from pathlib import Path

import pygame


WIDTH = 1280
HEIGHT = 820
BG = (18, 20, 24)
GRID = (54, 58, 64)
TEXT = (230, 234, 240)
MUTED = (145, 151, 160)
ARM = (86, 190, 255)
JOINT = (245, 247, 250)
GOAL = (80, 230, 150)
OBS = (255, 106, 106)
AXIS_X = (235, 87, 87)
AXIS_Y = (93, 201, 99)
AXIS_Z = (91, 144, 235)


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def shade(color, factor):
    factor = max(0.25, min(1.2, factor))
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def parse_trial_id(text):
    parts = text.strip().lower().split("-")
    if len(parts) != 4 or parts[0] not in {"n", "h"}:
        raise ValueError("format must be n/h-(obs)-(joints)-(trial), for example h-3-5-1")

    root = Path("figs_hard" if parts[0] == "h" else "figs")
    obs = int(parts[1])
    joints = int(parts[2])
    trial = int(parts[3])
    return root / f"obs_{obs}" / f"joints_{joints}" / f"trial_{trial:02d}"


def read_params(path):
    data = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for key, value in reader:
            data[key] = value
    return data


def parse_float_list(value):
    value = value.strip()
    if not value:
        return []
    return [float(x) for x in value.split()]


def load_trial(trial_dir):
    trial_dir = Path(trial_dir)
    params_path = trial_dir / "params.csv"
    links_path = trial_dir / "links.csv"
    obstacles_path = trial_dir / "obstacles.csv"

    if not params_path.exists() or not links_path.exists() or not obstacles_path.exists():
        raise FileNotFoundError(f"missing csv files under {trial_dir}")

    params = read_params(params_path)
    goal = tuple(parse_float_list(params["goal"]))
    rl = float(params.get("rl", "0.04"))

    obstacles = []
    with open(obstacles_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obstacles.append(
                (
                    (float(row["cx"]), float(row["cy"]), float(row["cz"])),
                    float(row["r"]),
                )
            )

    frames = []
    with open(links_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        point_count = (len(header) - 1) // 3
        for row in reader:
            pts = []
            for i in range(point_count):
                j = 1 + i * 3
                pts.append((float(row[j]), float(row[j + 1]), float(row[j + 2])))
            frames.append(pts)

    if not frames:
        raise ValueError(f"no link frames in {links_path}")

    return {
        "dir": trial_dir,
        "params": params,
        "goal": goal,
        "rl": rl,
        "obstacles": obstacles,
        "frames": frames,
    }


class Camera:
    def __init__(self, points, goal, obstacles):
        all_points = list(points) + [goal] + [c for c, _ in obstacles]
        self.target = tuple(sum(p[i] for p in all_points) / len(all_points) for i in range(3))
        span = 1.0
        for p in all_points:
            span = max(span, norm(vsub(p, self.target)))
        for c, r in obstacles:
            span = max(span, norm(vsub(c, self.target)) + r)

        self.distance = span * 3.0
        self.yaw = math.radians(42.0)
        self.pitch = math.radians(24.0)
        self.focal = 580.0
        self.near = 0.03

    def basis(self):
        cp = math.cos(self.pitch)
        offset = (
            self.distance * cp * math.cos(self.yaw),
            self.distance * cp * math.sin(self.yaw),
            self.distance * math.sin(self.pitch),
        )
        cam = vadd(self.target, offset)
        forward = unit(vsub(self.target, cam))
        right = unit(cross(forward, (0.0, 0.0, 1.0)))
        if norm(right) < 1e-9:
            right = (1.0, 0.0, 0.0)
        up = unit(cross(right, forward))
        return cam, right, up, forward

    def project(self, point):
        cam, right, up, forward = self.basis()
        rel = vsub(point, cam)
        x = dot(rel, right)
        y = dot(rel, up)
        z = dot(rel, forward)
        if z <= self.near:
            return None
        sx = WIDTH * 0.5 + self.focal * x / z
        sy = HEIGHT * 0.5 - self.focal * y / z
        return sx, sy, z

    def orbit(self, dx, dy):
        self.yaw -= dx * 0.008
        self.pitch += dy * 0.008
        limit = math.radians(86.0)
        self.pitch = max(-limit, min(limit, self.pitch))

    def pan_screen(self, dx, dy):
        _, right, up, _ = self.basis()
        scale = self.distance / self.focal
        self.target = vadd(self.target, vadd(vscale(right, -dx * scale), vscale(up, dy * scale)))

    def move_local(self, forward_amount, right_amount, up_amount):
        _, right, up, forward = self.basis()
        flat_forward = unit((forward[0], forward[1], 0.0))
        step = self.distance * 0.018
        delta = (0.0, 0.0, 0.0)
        delta = vadd(delta, vscale(flat_forward, forward_amount * step))
        delta = vadd(delta, vscale(right, right_amount * step))
        delta = vadd(delta, vscale(up, up_amount * step))
        self.target = vadd(self.target, delta)

    def zoom(self, factor):
        self.distance = max(0.2, min(25.0, self.distance * factor))


def draw_text(screen, font, x, y, text, color=TEXT):
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))


def draw_grid(screen, camera):
    lines = []
    size = 2.5
    step = 0.25
    count = int(size / step)
    for i in range(-count, count + 1):
        v = i * step
        color = (72, 76, 82) if i == 0 else GRID
        lines.append(((v, -size, 0.0), (v, size, 0.0), color))
        lines.append(((-size, v, 0.0), (size, v, 0.0), color))

    axes = [
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), AXIS_X),
        ((0.0, 0.0, 0.0), (0.0, 1.2, 0.0), AXIS_Y),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.2), AXIS_Z),
    ]

    for a, b, color in lines + axes:
        pa = camera.project(a)
        pb = camera.project(b)
        if pa and pb:
            pygame.draw.line(screen, color, (pa[0], pa[1]), (pb[0], pb[1]), 1)


def add_sphere(primitives, camera, center, radius, color, alpha, rings=False):
    projected = camera.project(center)
    if not projected:
        return
    sx, sy, depth = projected
    pr = max(3, int(camera.focal * radius / depth))
    primitives.append(("sphere", depth, sx, sy, pr, color, alpha, rings))


def add_link(primitives, camera, a, b, radius, color):
    pa = camera.project(a)
    pb = camera.project(b)
    if not pa or not pb:
        return
    depth = (pa[2] + pb[2]) * 0.5
    width = max(2, int(camera.focal * radius * 2.2 / depth))
    primitives.append(("line", depth, pa[0], pa[1], pb[0], pb[1], width, color))


def draw_primitives(screen, primitives):
    for prim in sorted(primitives, key=lambda p: p[1], reverse=True):
        if prim[0] == "line":
            _, depth, x1, y1, x2, y2, width, color = prim
            pygame.draw.line(screen, shade(color, 1.15 - depth * 0.05), (x1, y1), (x2, y2), width)
        elif prim[0] == "sphere":
            _, depth, x, y, radius, color, alpha, rings = prim
            local = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            center = (radius + 2, radius + 2)
            fill = (*shade(color, 1.1 - depth * 0.04), alpha)
            pygame.draw.circle(local, fill, center, radius)
            screen.blit(local, (int(x - radius - 2), int(y - radius - 2)))
            pygame.draw.circle(screen, shade(color, 1.2), (int(x), int(y)), radius, 1)
            if rings:
                pygame.draw.circle(screen, shade(color, 0.9), (int(x), int(y)), radius, 1)
                pygame.draw.ellipse(
                    screen,
                    shade(color, 0.75),
                    (int(x - radius), int(y - radius * 0.35), radius * 2, max(2, radius * 0.7)),
                    1,
                )


def run_viewer(trial):
    pygame.init()
    pygame.display.set_caption(f"3D replay: {trial['dir']}")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 15)

    frames = trial["frames"]
    camera = Camera(frames[0], trial["goal"], trial["obstacles"])
    step_index = 0
    playing = False
    accumulator = 0.0
    step_hold_timer = 0.0
    step_hold_dir = 0
    dragging = None
    last_mouse = (0, 0)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_COMMA:
                    step_index = max(0, step_index - 1)
                    playing = False
                    step_hold_timer = 0.0
                elif event.key == pygame.K_PERIOD:
                    step_index = min(len(frames) - 1, step_index + 1)
                    playing = False
                    step_hold_timer = 0.0
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_HOME:
                    camera = Camera(frames[step_index], trial["goal"], trial["obstacles"])
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = "orbit"
                    last_mouse = event.pos
                elif event.button == 3:
                    dragging = "pan"
                    last_mouse = event.pos
                elif event.button == 4:
                    camera.zoom(0.88)
                elif event.button == 5:
                    camera.zoom(1.14)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in {1, 3}:
                    dragging = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                last_mouse = event.pos
                if dragging == "orbit":
                    camera.orbit(dx, dy)
                else:
                    camera.pan_screen(dx, dy)

        keys = pygame.key.get_pressed()
        held_dir = (1 if keys[pygame.K_PERIOD] else 0) - (1 if keys[pygame.K_COMMA] else 0)
        if held_dir != 0:
            playing = False
            if held_dir != step_hold_dir:
                step_hold_dir = held_dir
                step_hold_timer = 0.0
            else:
                step_hold_timer += dt
                while step_hold_timer >= 0.035:
                    step_hold_timer -= 0.035
                    step_index = max(0, min(len(frames) - 1, step_index + held_dir))
        else:
            step_hold_dir = 0
            step_hold_timer = 0.0

        camera.move_local(
            (1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0),
            (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0),
            (1 if keys[pygame.K_e] else 0) - (1 if keys[pygame.K_q] else 0),
        )

        if playing:
            accumulator += dt
            if accumulator >= 1.0 / 24.0:
                accumulator = 0.0
                step_index = min(len(frames) - 1, step_index + 1)
                if step_index == len(frames) - 1:
                    playing = False

        screen.fill(BG)
        draw_grid(screen, camera)

        primitives = []
        for center, radius in trial["obstacles"]:
            add_sphere(primitives, camera, center, radius, OBS, 58, rings=True)

        pts = frames[step_index]
        for a, b in zip(pts, pts[1:]):
            add_link(primitives, camera, a, b, trial["rl"], ARM)
        for p in pts:
            add_sphere(primitives, camera, p, trial["rl"] * 1.25, JOINT, 255)
        add_sphere(primitives, camera, trial["goal"], trial["rl"] * 2.0, GOAL, 240)
        draw_primitives(screen, primitives)

        title = f"{trial['dir']}    step {step_index}/{len(frames) - 1}"
        draw_text(screen, font, 16, 14, title)
        draw_text(screen, small, 16, HEIGHT - 54, "mouse drag: orbit | right drag: pan | wheel: zoom | WASD/QE: move | , .: step | space: play | home: reset | esc: quit", MUTED)
        draw_text(screen, small, 16, HEIGHT - 30, "red: obstacles | green: goal | cyan: arm", MUTED)

        pygame.display.flip()

    pygame.quit()


def main():
    if len(sys.argv) >= 2:
        text = sys.argv[1]
    else:
        text = input("trial id (ex: h-3-5-1): ")

    trial_dir = parse_trial_id(text)
    trial = load_trial(trial_dir)
    run_viewer(trial)


if __name__ == "__main__":
    main()