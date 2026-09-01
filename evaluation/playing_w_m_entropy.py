

import numpy as np

print("\n\n")
print("Confidently correct")
x = .999
y = (1 - x)/9
m_entropy = - (x * np.log(x)) - 9 * y * np.log(1 - y)
print(f"{m_entropy:.4f}")


print("Not confident in any direction")
x = .1
y = (1 - x)/9
m_entropy = - (x * np.log(x)) - 9 * y * np.log(1 - y)
print(f"{m_entropy:.4f}")


print("Confidently wrong")
x_next = .999999999999
x_remaining = (1 - x_next)/9
m_entropy = - (x_remaining * np.log(x_remaining)) - (8 * x_remaining * np.log(1 - x_remaining) + x_next * np.log(1 - x_next))
print(f"{m_entropy:.4f}")

print("\n\n")