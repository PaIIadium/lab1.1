import numpy as np
import matplotlib.pyplot as plp

W_MAX = 2500
HARMONICS_QUANTITY = 14
DISCRETE_SAMPLES_NUMBER = 64


def calculate_value(amplitude, phase, frequency, time):
    return amplitude * np.sin(frequency * time + phase)


def create_signal(w_max, harmonics_quantity, discrete_samples_number):
    sum_values = np.zeros(discrete_samples_number)
    sampling_frequency = 2 * w_max
    for w in range(1, harmonics_quantity + 1):
        phase = np.random.uniform(-np.pi / 2, np.pi / 2)
        amplitude = np.random.uniform(0.0, 1000.0)
        frequency = 2 * np.pi * w_max / harmonics_quantity * w
        for t in range(discrete_samples_number):
            sum_values[t] += calculate_value(amplitude, phase, frequency, t / sampling_frequency)
    return sum_values


def calc_w_pkn(p, k, n):
    return complex(np.cos(2 * np.pi / n * p * k), np.sin(2 * np.pi / n * p * k))


def discrete_fourier_transform(signal):
    result = np.zeros(len(signal), dtype=np.complex)
    wpk_table = np.zeros((len(signal), len(signal)), dtype=np.complex)
    for p in range(len(signal)):
        for k in range(len(signal)):
            w = calc_w_pkn(p, k, len(signal))
            wpk_table[p][k] = w
            result[p] += w * signal[k]
    for p in range(len(wpk_table)):
        print(p, wpk_table[p])
    return abs(result)


SIGNAL = create_signal(W_MAX, HARMONICS_QUANTITY, DISCRETE_SAMPLES_NUMBER)

dft = discrete_fourier_transform(SIGNAL)
fft = abs(np.fft.fft(SIGNAL))

time_period = DISCRETE_SAMPLES_NUMBER / (2 * W_MAX)

plp.subplot(221)
plp.title('Вихідний сигнал')
plp.plot(np.linspace(0, time_period, num=DISCRETE_SAMPLES_NUMBER), SIGNAL)

plp.subplot(223)
plp.title('Дискретне перетворення Фур\'є (частотний спектр)')
plp.plot(np.linspace(0, 2 * W_MAX, num=DISCRETE_SAMPLES_NUMBER), dft)

plp.subplot(224)
plp.title('Швидке перетворення Фур\'є (з бібліотеки numpy)')
plp.plot(np.linspace(0, 2 * W_MAX, num=DISCRETE_SAMPLES_NUMBER), fft)

plp.show()
