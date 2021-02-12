import numpy as np
import matplotlib.pyplot as plp

w_max = 2500
harmonics_quantity = 14
discrete_samples_number = 64

def calculate(amplitude, phase, frequency, time):
    return amplitude * np.sin(frequency * time + phase)

sum_values = np.zeros(discrete_samples_number)

for w in range(1, harmonics_quantity + 1):
    phase = np.random.uniform(-np.pi / 2, np.pi / 2)
    amplitude = np.random.uniform(0.0, 1000.0)
    for t in range(discrete_samples_number):
        sum_values[t] += calculate(amplitude, phase, w_max / harmonics_quantity * w, t)

print('Average:', np.average(sum_values))
print('Dispersion:', np.std(sum_values) ** 2)

plp.plot(sum_values)
plp.show()
