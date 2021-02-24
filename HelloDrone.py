#import setup_path 
import airsim


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.simPrintLogMessage("Ready to Operate")