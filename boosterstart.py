import booster

client = booster.BoosterClient("127.0.0.1")
status = client.get_robot_status()

print(f"Mode:    {status.mode}")
print(f"Battery: {status.battery_percentage:.1f}%")
print(f"IMU:     {status.imu_status}")