import carla, random, time, os, json, argparse
import numpy as np
import queue
from tqdm import tqdm
import os

# --- CONFIGURATION ---
NUM_RUNS = 3
RUN_DURATION = 600      # seconds per run
IMG_WIDTH = 800
IMG_HEIGHT = 600
NUM_NPC_VEHICLES = 50
NUM_NPC_WALKERS = 30
SAVE_EVERY_N = 5        # Save every N-th frame (for all cameras)
OUTPUT_ROOT = os.path.join(
    os.environ.get("CARLA_DATA_PATH", os.path.expanduser("~")), "automoe_training"
)

CAMERA_CONFIGS = [
    ("front",       carla.Transform(carla.Location(x=1.5, z=2.4))),
    ("front_left",  carla.Transform(carla.Location(x=1.2, y=-0.5, z=2.2), carla.Rotation(yaw=-45))),
    ("front_right", carla.Transform(carla.Location(x=1.2, y= 0.5, z=2.2), carla.Rotation(yaw=45))),
    ("rear",        carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180))),
]

def find_weather_presets():
    return [getattr(carla.WeatherParameters, name) for name in dir(carla.WeatherParameters)
            if name[0].isupper() and isinstance(getattr(carla.WeatherParameters, name), carla.WeatherParameters)]

def parse_args():
    parser = argparse.ArgumentParser(description='CARLA Autopilot Data Collection')
    parser.add_argument('--runs', type=int, default=NUM_RUNS)
    parser.add_argument('--duration', type=int, default=RUN_DURATION)
    parser.add_argument('--vehicles', type=int, default=NUM_NPC_VEHICLES)
    parser.add_argument('--walkers', type=int, default=NUM_NPC_WALKERS)
    parser.add_argument('--output', type=str, default=OUTPUT_ROOT)
    parser.add_argument('--single-run', action='store_true')
    parser.add_argument('--continue-from', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=SAVE_EVERY_N, help="Save every N-th frame")
    return parser.parse_args()

def batch_destroy(client, actors):
    cmds = [carla.command.DestroyActor(x) for x in actors if x.is_alive]
    if cmds:
        client.apply_batch(cmds)

def get_latest_from_queue(q):
    img = None
    while True:
        try:
            img = q.get_nowait()
        except queue.Empty:
            break
    return img

def main():
    args = parse_args()
    num_runs = 1 if args.single_run else args.runs
    run_duration = args.duration
    num_npc_vehicles = args.vehicles
    num_npc_walkers = args.walkers
    start_run = args.continue_from
    save_every_n = args.save_every
    output_root = args.output

    print(f"🚗 CARLA Data Collection Starting...")
    print(f"   Runs: {num_runs} (starting from #{start_run})")
    print(f"   Duration: {run_duration/60:.1f} minutes per run")
    print(f"   NPCs: {num_npc_vehicles} vehicles, {num_npc_walkers} walkers")
    print(f"   Output: {output_root}\n")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    # --- Enable synchronous mode globally for safety ---
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    # --- Batch cleanup lingering vehicles ---
    print("🧹 Cleaning up existing vehicles...")
    lingering_vehicles = [actor for actor in world.get_actors().filter('vehicle.*')]
    batch_destroy(client, lingering_vehicles)
    time.sleep(1)

    os.makedirs(output_root, mode=0o755, exist_ok=True)
    weather_presets = find_weather_presets()
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('role_name', 'ego')

    for run_i in range(start_run, start_run + num_runs):
        run_id = f"run_{run_i:03d}"
        run_dir = os.path.join(output_root, run_id)
        for cam_name, _ in CAMERA_CONFIGS:
            os.makedirs(os.path.join(run_dir, "images", cam_name), mode=0o755, exist_ok=True)

        # -- Weather and NPCs --
        weather = random.choice(weather_presets)
        weather.sun_altitude_angle = random.uniform(-90, 90)
        world.set_weather(weather)

        # -- Spawn NPC vehicles --
        spawn_points = world.get_map().get_spawn_points()
        available_spawns = spawn_points.copy()
        random.shuffle(available_spawns)
        vehicle_bps = blueprint_library.filter('vehicle.*')
        npc_vehicles = []
        for i in range(min(num_npc_vehicles, len(available_spawns) - 10)):
            try:
                npc_bp = random.choice(vehicle_bps)
                if 'tesla.model3' in npc_bp.id:
                    continue
                spawn_point = available_spawns[i]
                npc_vehicle = world.spawn_actor(npc_bp, spawn_point)
                if npc_vehicle is not None:
                    npc_vehicles.append(npc_vehicle)
                    npc_vehicle.set_autopilot(True, tm.get_port())
            except Exception:
                continue
        remaining_spawns = available_spawns[len(npc_vehicles):]

        # -- Spawn ego vehicle --
        spawn_pt = random.choice(remaining_spawns)
        vehicle = world.spawn_actor(vehicle_bp, spawn_pt)
        vehicle.set_autopilot(True, tm.get_port())

        # -- Spawn pedestrians if desired --
        npc_walkers = []
        if num_npc_walkers > 0:
            walker_bps = blueprint_library.filter('walker.pedestrian.*')
            walker_spawn_points = []
            for i in range(num_npc_walkers):
                spawn_point = carla.Transform()
                loc = world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc
                    walker_spawn_points.append(spawn_point)
            for spawn_point in walker_spawn_points:
                try:
                    walker_bp = random.choice(walker_bps)
                    walker = world.spawn_actor(walker_bp, spawn_point)
                    if walker is not None:
                        npc_walkers.append(walker)
                except Exception:
                    continue

        # --- Synchronous Sensor Queues for Each Camera ---
        sensor_queues = {cam_name: queue.Queue() for cam_name in dict(CAMERA_CONFIGS).keys()}

        # -- Set up all cameras and register their queues --
        cameras = {}
        for cam_name, cam_tf in CAMERA_CONFIGS:
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(IMG_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
            camera_bp.set_attribute('fov', '90')
            cam_actor = world.spawn_actor(camera_bp, cam_tf, attach_to=vehicle)
            cameras[cam_name] = cam_actor
            # Each camera puts its data in its respective queue
            cam_actor.listen(sensor_queues[cam_name].put)

        # -- Set up collision sensor --
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_events = []

        def collision_callback(event):
            collision_events.append({
                "timestamp": event.timestamp,
                "actor_type": event.other_actor.type_id,
                "actor_id": event.other_actor.id,
                "impulse": {"x": event.normal_impulse.x, "y": event.normal_impulse.y, "z": event.normal_impulse.z}
            })
            print(f"⚠️ [{run_id}] Collision detected at {event.timestamp} with {event.other_actor.type_id}")

        collision_sensor.listen(collision_callback)

        frame_data = []
        start_time = time.time()

        print(f"[{run_id}] Started: map {world.get_map().name}, weather {weather}")
        print(f"[{run_id}] Traffic: {len(npc_vehicles)} vehicles, {len(npc_walkers)} pedestrians")
        print(f"[{run_id}] All cameras enabled (saving every {save_every_n} frames, **synchronized**).")

        # --- Tick the simulator and synchronously save all camera images every N ticks ---
        num_ticks = int(run_duration / 0.05)
        for tick in tqdm(range(num_ticks), desc=f"[{run_id}] Collecting data"):
            world.tick()

            # Drain all camera queues every tick, keep only latest image per camera
            latest_imgs = {}
            for cam_name in cameras.keys():
                latest_imgs[cam_name] = get_latest_from_queue(sensor_queues[cam_name])

            if tick % save_every_n == 0:
                frame_id = tick  # For consistency, use tick as frame_id
                fname = f"{frame_id:06d}.png"
                for cam_name, img in latest_imgs.items():
                    if img is not None:
                        img_path = os.path.join(run_dir, "images", cam_name, fname)
                        img.save_to_disk(img_path)
                    else:
                        print(f"⚠️ [{run_id}] Warning: No image from {cam_name} at tick {tick}")

                # Save metadata (using front camera image timestamp for simplicity)
                tf = vehicle.get_transform()
                vel = vehicle.get_velocity()
                ctrl = vehicle.get_control()
                nearby_vehicles = world.get_actors().filter('vehicle.*')
                nearby_count = len([v for v in nearby_vehicles if v.get_location().distance(tf.location) < 50.0])
                frame_info = {
                    "frame": frame_id,
                    "timestamp": time.time() - start_time,
                    "image_filename": fname,
                    "location": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z},
                    "rotation": {"pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw, "roll": tf.rotation.roll},
                    "velocity": {"x": vel.x, "y": vel.y, "z": vel.z},
                    "speed_kmh": 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2),
                    "control": {"throttle": ctrl.throttle, "steer": ctrl.steer, "brake": ctrl.brake},
                    "traffic_density": {
                        "nearby_vehicles_50m": nearby_count - 1,
                        "total_npc_vehicles": len(npc_vehicles),
                        "total_npc_walkers": len(npc_walkers)
                    }
                }
                frame_data.append(frame_info)

        # --- Cleanup (batch destroy for all) ---
        for cam in cameras.values():
            cam.stop()
        collision_sensor.stop()
        actors_to_destroy = [vehicle, collision_sensor] + list(cameras.values()) + npc_vehicles + npc_walkers
        batch_destroy(client, actors_to_destroy)

        # --- Save logs ---
        if frame_data:
            avg_speed = np.mean([f["speed_kmh"] for f in frame_data])
            steering_variance = np.var([f["control"]["steer"] for f in frame_data])
            avg_traffic = np.mean([f["traffic_density"]["nearby_vehicles_50m"] for f in frame_data])
        else:
            avg_speed = steering_variance = avg_traffic = 0

        quality_metrics = {
            "avg_speed_kmh": avg_speed,
            "steering_variance": steering_variance,
            "collision_count": len(collision_events),
            "avg_nearby_traffic": avg_traffic,
            "weather_difficulty": weather.precipitation + getattr(weather, "fog_density", 0),
            "frames_collected": len(frame_data)
        }

        with open(os.path.join(run_dir, "vehicle_log.json"), "w") as f:
            json.dump(frame_data, f, indent=2)
        with open(os.path.join(run_dir, "collisions.json"), "w") as f:
            json.dump(collision_events, f, indent=2)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({
                "run_id": run_id,
                "map": world.get_map().name,
                "weather": {
                    "preset": str(weather),
                    "sun_altitude_angle": weather.sun_altitude_angle,
                    "cloudiness": weather.cloudiness,
                    "precipitation": weather.precipitation,
                    "wetness": weather.wetness,
                    "fog_density": getattr(weather, "fog_density", 0)
                },
                "spawn_point": {"x": spawn_pt.location.x, "y": spawn_pt.location.y, "z": spawn_pt.location.z},
                "timestamp_start": start_time,
                "duration_seconds": run_duration,
                "all_cameras_enabled": True,
                "save_every_n": save_every_n,
                "npc_config": {
                    "vehicles_spawned": len(npc_vehicles),
                    "walkers_spawned": len(npc_walkers)
                },
                "quality_metrics": quality_metrics
            }, f, indent=2)

        print(f"[{run_id}] Complete—{len(frame_data)} frames, {len(collision_events)} collisions, quality score: {avg_speed:.1f} km/h avg")

    # -- Restore async mode for safety --
    settings.synchronous_mode = False
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)
    print("Finished all runs and restored async mode.")

if __name__ == "__main__":
    main()
