import numpy as np
import math

dist = np.array([10,0,0])
a = np.array([5,0,0])
b = np.array([0,0,0])
c = np.array([-10,0,0])


dist_a = np.linalg.norm(dist-a)
dist_b = np.linalg.norm(dist-b)
dist_c = np.linalg.norm(dist-c)

print("dis")
print("dis_a ", dist_a)
print("dis_b ", dist_b)
print("dis_c ", dist_c)
print("")


print("math.exp")
print("a ", math.exp(dist_a))
print("b ", math.exp(dist_b))
print("c ", math.exp(dist_c))
print("")


beta = 1

print("beta")
print("a ", math.exp(-beta * dist_a) - 0.5)
print("b ", math.exp(-beta * dist_b) - 0.5)
print("c ", math.exp(-beta * dist_c) - 0.5)
print("")

print("1/dist_x")
print("a ", 10/dist_a + 10)
print("b ", 10/dist_b + 10)
print("c ", 10/dist_c + 10)
print("")


reward_speed = np.linalg.norm([4,0,3])
print("reward_speed ", reward_speed)
print("")