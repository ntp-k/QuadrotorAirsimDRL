#import setup_path 
from airsim.AirSimClient import *


# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

landed = client.getMultirotorState().landed_state
if landed == LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    print("already flying...")
    client.hoverAsync().join()

print("flying test...")
#NED coordinate system, +X is North, +Y is East and +Z is Down
client.moveToPositionAsync(0,0,-10,5).join()
client.moveToPositionAsync(10,0,-10,5).join()
client.moveToPositionAsync(-10,0,-10,5).join()
client.moveToPositionAsync(0,0,-10,5).join()
client.moveToPositionAsync(0,-10,-10,5).join()
client.moveToPositionAsync(0,10,-10,5).join()
client.moveToPositionAsync(0,0,-10,5).join()
client.hoverAsync().join()

print("done")

