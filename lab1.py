import numpy as np
import matplotlib.pyplot as plp
import time

w_max = 2500
harmonics_quantity = 14
discrete_samples_number = 64


def calculate(amplitude, phase, frequency, time):
    return amplitude * np.sin(frequency * time + phase)


sum_values = np.zeros(discrete_samples_number)

harmonics_times = dict()

for w in range(1, harmonics_quantity + 1):
    begin_time = time.perf_counter()
    phase = np.random.uniform(-np.pi / 2, np.pi / 2)
    frequency = w_max / harmonics_quantity * w
    amplitude = np.random.uniform(0.0, 1000.0)
    for t in range(discrete_samples_number):
        sum_values[t] += calculate(amplitude, phase, frequency, t)
    end_time = time.perf_counter()
    harmonics_times[str(round(frequency)) + ' Hz'] = str(round((end_time - begin_time) * 1_000_000)) + ' us'

print('Average:', np.average(sum_values))
print('Dispersion:', np.std(sum_values) ** 2)
print('Harmonics generation time:', str(harmonics_times))

plp.plot(sum_values)
plp.show()
