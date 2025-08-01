import carla

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    print("✅ CARLA connection successful despite chmod error!")
    print("Available maps:", client.get_available_maps())
except Exception as e:
    print(f"❌ CARLA connection failed: {e}")
