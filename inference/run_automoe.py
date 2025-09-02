import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image

try:
    import carla  # type: ignore
except Exception as e:  # pragma: no cover
    carla = None
    print("Warning: CARLA Python API not found.")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.automoe import create_automoe_model  # noqa: E402


def build_image_transform(target_hw: Tuple[int, int] = (256, 256)) -> nn.Module:
    return T.Compose([
        T.ToPILImage(),
        T.Resize(target_hw, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def model_infer(model: nn.Module,
                image_rgb: np.ndarray,
                last_speed_kmh: float,
                device: torch.device,
                img_tf: nn.Module) -> Dict[str, torch.Tensor]:
    # image_rgb: [H,W,3] uint8
    tensor = img_tf(image_rgb).unsqueeze(0).to(device)  # [1,3,H,W]
    speed = torch.tensor([[last_speed_kmh]], dtype=torch.float32, device=device)  # [1,1]
    batch: Dict[str, Any] = {
        'image': tensor,
        'speed': speed,
        # Controls not available at inference → zeros (simple context)
        'steering': torch.zeros(1, 1, device=device),
        'throttle': torch.zeros(1, 1, device=device),
        'brake': torch.zeros(1, 1, device=device),
    }
    with torch.autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'), enabled=True):
        pred = model(batch)
    return pred


class PID:
    def __init__(self, kp: float, ki: float, kd: float, clamp: Tuple[float, float] = (0.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.clamp_min, self.clamp_max = clamp
        self.integral = 0.0
        self.prev_err: Optional[float] = None

    def reset(self):
        self.integral = 0.0
        self.prev_err = None

    def step(self, err: float, dt: float) -> float:
        self.integral += err * dt
        deriv = 0.0 if self.prev_err is None else (err - self.prev_err) / max(dt, 1e-3)
        self.prev_err = err
        out = self.kp * err + self.ki * self.integral + self.kd * deriv
        return float(np.clip(out, self.clamp_min, self.clamp_max))


def pure_pursuit_steer(waypoints_xy: np.ndarray, lookahead_m: float = 3.0, wheel_base_m: float = 2.8) -> float:
    # waypoints in ego frame: x right, y forward, shape [H,2]
    if waypoints_xy.size == 0:
        return 0.0
    dists = np.linalg.norm(waypoints_xy, axis=1)
    idx = int(np.argmin(np.abs(dists - lookahead_m)))
    target = waypoints_xy[idx]
    x, y = float(target[0]), float(target[1])
    if y <= 1e-3:
        return 0.0
    curvature = (2.0 * x) / (y * y + x * x)
    steer = math.atan(wheel_base_m * curvature)
    return float(np.clip(steer, -1.0, 1.0))


def carla_image_to_rgb(image) -> np.ndarray:
    # CARLA camera returns BGRA uint8 buffer
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]
    rgb = bgr[:, :, ::-1].copy()
    return rgb


def to_finite_float(value: float, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return float(default)


def setup_carla_world(client, town: Optional[str], fixed_delta_seconds: float):
    world = client.get_world()
    if town and world.get_map().name != town:
        world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    return world


def spawn_ego_vehicle(world, blueprint_lib, spawn_point=None):
    bp = blueprint_lib.find('vehicle.tesla.model3')
    bp.set_attribute('role_name', 'ego')
    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    return vehicle


def attach_camera(world, blueprint_lib, vehicle, width: int, height: int, fov: float):
    cam_bp = blueprint_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(width))
    cam_bp.set_attribute('image_size_y', str(height))
    cam_bp.set_attribute('fov', str(fov))
    # Position camera at hood
    transform = carla.Transform(carla.Location(x=1.5, z=1.6))
    camera = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
    return camera


def load_model(model_config_path: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    cfg = json.loads(Path(model_config_path).read_text())
    model = create_automoe_model(cfg, device)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get('model_state_dict', state)
    # Strip DDP prefixes if present
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Loaded with relaxed matching. Missing={len(missing)} Unexpected={len(unexpected)}")
    model.eval()
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inference: Run AutoMoE in CARLA closed-loop (skeleton)")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--client_timeout', type=float, default=30.0)
    parser.add_argument('--town', type=str, default=None)
    parser.add_argument('--fixed_dt', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--fov', type=float, default=90.0)
    parser.add_argument('--model_config', type=str, default='models/configs/automoe/model_config.json')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='inference/results')
    parser.add_argument('--default_start_kmh', type=float, default=20.0,
                        help='Fallback target speed used during bootstrap or when model speed is unavailable')
    parser.add_argument('--bootstrap_steps', type=int, default=20,
                        help='Number of steps to apply conservative bootstrap logic to start moving')
    parser.add_argument('--lookahead_m', type=float, default=3.0)
    parser.add_argument('--kp', type=float, default=0.4)
    parser.add_argument('--ki', type=float, default=0.0)
    parser.add_argument('--kd', type=float, default=0.02)
    parser.add_argument('--save_frames', action='store_true', help='Save RGB frames to out_dir/frames')
    parser.add_argument('--record_every', type=int, default=1, help='Save every Nth frame')
    parser.add_argument('--export_gif', action='store_true', help='Export an animated GIF from saved frames')
    parser.add_argument('--gif_fps', type=int, default=10, help='Frames per second for GIF export')
    parser.add_argument('--gif_sample_every', type=int, default=1, help='Sample every Nth saved frame when building GIF')
    parser.add_argument('--gif_max_frames', type=int, default=800, help='Hard cap on number of frames in GIF')
    args = parser.parse_args()

    if carla is None:
        print("CARLA API not available. Exiting.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create a per-run directory to partition tries
    ts = int(time.time())
    run_dir = out_dir / f'run_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / 'frames'
    if args.save_frames or args.export_gif:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = load_model(args.model_config, args.checkpoint, device)
    img_tf = build_image_transform((256, 256))
    speed_pid = PID(args.kp, args.ki, args.kd, clamp=(0.0, 1.0))

    # Connect CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(float(args.client_timeout))
    world = setup_carla_world(client, args.town, args.fixed_dt)
    bp_lib = world.get_blueprint_library()

    vehicle = None
    camera = None
    latest_image = {'frame': -1, 'rgb': None}

    try:
        vehicle = spawn_ego_vehicle(world, bp_lib)
        camera = attach_camera(world, bp_lib, vehicle, args.width, args.height, args.fov)

        def _on_image(image):
            latest_image['frame'] = image.frame
            latest_image['rgb'] = carla_image_to_rgb(image)

        camera.listen(_on_image)

        # Warmup: tick until first camera frame arrives (up to ~60 ticks)
        got_first = False
        for _ in range(60):
            world.tick()
            if latest_image['rgb'] is not None:
                got_first = True
                break
        if not got_first:
            print("Warning: No camera frames received during warmup.")

        logs = []
        last_speed_kmh = 0.0
        for step in range(args.steps):
            world.tick()
            rgb = latest_image['rgb']
            if rgb is None:
                control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                vehicle.apply_control(control)
                continue

            pred = model_infer(model, rgb, last_speed_kmh, device, img_tf)

            # Waypoints are ego-frame [H,2] in meters
            waypoints = pred['waypoints'].detach().cpu().numpy()[0]  # [H,2]
            if waypoints.size == 0 or not np.isfinite(waypoints).all():
                steer = 0.0
            else:
                steer = pure_pursuit_steer(waypoints, lookahead_m=args.lookahead_m)

            # Target speed from model; robust extraction with bootstrap fallback
            target_kmh = float('nan')
            spd_out = pred.get('speed')
            if isinstance(spd_out, torch.Tensor) and spd_out.numel() > 0:
                try:
                    target_kmh = float(spd_out.detach().flatten()[-1].item())
                except Exception:
                    target_kmh = float('nan')
            if not math.isfinite(target_kmh):
                # During initial steps or when speed not predicted, use a reasonable default to get moving
                target_kmh = args.default_start_kmh if step < int(args.bootstrap_steps) else last_speed_kmh
            # Read vehicle speed (m/s) from velocity vector
            vel = vehicle.get_velocity()
            curr_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            curr_kmh = curr_ms * 3.6
            last_speed_kmh = curr_kmh

            err_kmh = max(0.0, target_kmh) - curr_kmh
            throttle_cmd = speed_pid.step(err_kmh, args.fixed_dt)
            # Bootstrap: if still stationary, nudge throttle to overcome stiction
            if step < int(args.bootstrap_steps) and curr_kmh < 1.0 and throttle_cmd < 0.2:
                throttle_cmd = max(throttle_cmd, 0.3)
            # Ensure finite numeric commands
            steer = to_finite_float(steer, 0.0)
            throttle_cmd = to_finite_float(throttle_cmd, 0.0)
            brake_cmd = 0.0 if err_kmh >= 0.0 else min(1.0, -err_kmh / 20.0)

            control = carla.VehicleControl(
                throttle=float(np.clip(throttle_cmd, 0.0, 1.0)),
                steer=float(np.clip(steer, -1.0, 1.0)),
                brake=float(np.clip(brake_cmd, 0.0, 1.0)),
            )
            vehicle.apply_control(control)

            # Optional: save frames
            if (args.save_frames or args.export_gif) and (step % args.record_every == 0):
                try:
                    frame_path = frames_dir / f'frame_{step:06d}.jpg'
                    Image.fromarray(rgb).save(frame_path, format='JPEG', quality=90)
                except Exception:
                    # Non-fatal; continue driving
                    pass

            # Minimal logging per step
            logs.append({
                'step': int(step),
                'speed_kmh': float(curr_kmh),
                'target_speed_kmh': float(target_kmh),
                'steer': to_finite_float(control.steer, 0.0),
                'throttle': to_finite_float(control.throttle, 0.0),
                'brake': to_finite_float(control.brake, 0.0),
            })

        # Persist logs
        (run_dir / 'log.json').write_text(json.dumps(logs, indent=2))
        print(f"Saved logs to {run_dir}/log.json")

        # Optional: export GIF from saved frames
        if args.export_gif:
            try:
                # Prefer streaming writer if imageio is available to avoid high memory usage
                try:
                    import imageio.v2 as imageio  # type: ignore
                    gif_path = run_dir / f'run_{ts}.gif'
                    print(f"Exporting GIF to {gif_path} (fps={args.gif_fps})...")
                    written = 0
                    with imageio.get_writer(gif_path, mode='I', duration=max(1e-6, 1.0/float(args.gif_fps))) as writer:
                        frame_files = sorted(frames_dir.glob('frame_*.jpg'))
                        for idx, fp in enumerate(frame_files):
                            if (idx % max(1, args.gif_sample_every)) != 0:
                                continue
                            writer.append_data(imageio.imread(fp))
                            written += 1
                            if written >= max(1, args.gif_max_frames):
                                break
                    print(f"GIF saved: {gif_path} (frames written: {written})")
                except Exception:
                    # Fallback to PIL-based GIF creation (may use more memory)
                    from PIL import Image as _PILImage
                    gif_path = run_dir / f'run_{ts}.gif'
                    print(f"Exporting GIF via PIL to {gif_path} (fps={args.gif_fps})...")
                    frame_files = sorted(frames_dir.glob('frame_*.jpg'))
                    selected = []
                    for idx, fp in enumerate(frame_files):
                        if (idx % max(1, args.gif_sample_every)) != 0:
                            continue
                        selected.append(fp)
                        if len(selected) >= max(1, args.gif_max_frames):
                            break
                    if not selected:
                        print("No frames available for GIF export.")
                    else:
                        imgs = []
                        for fp in selected:
                            img = _PILImage.open(fp).convert('RGB')
                            imgs.append(img)
                        duration_ms = int(1000.0 / float(args.gif_fps))
                        imgs[0].save(
                            gif_path,
                            save_all=True,
                            append_images=imgs[1:],
                            duration=duration_ms,
                            loop=0,
                            format='GIF'
                        )
                        print(f"GIF saved: {gif_path} (frames used: {len(imgs)})")
            except Exception as gif_e:
                print(f"GIF export failed: {gif_e}")

    finally:
        # Cleanup actors and restore settings
        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
            if vehicle is not None:
                vehicle.destroy()
        except Exception:
            pass
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass


if __name__ == '__main__':
    main()


