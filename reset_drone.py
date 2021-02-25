from airsim.AirSimClient import *

import time

# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("fly")
client.moveToPositionAsync(0, 0, -2, 5).join()

print("reset")
client.reset()
time.sleep(2)


print("start")
client.enableApiControl(True)
client.armDisarm(True)
client.moveToPositionAsync(0, 0, -1, 5).join()
client.landAsync().join()

print("done")
