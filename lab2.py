import time

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
    return result


twiddle_factors = None


def calculate_twiddle_factors(k):
    global twiddle_factors
    twiddle_factors = dict()
    size = 1
    while size <= k:
        twiddle_factors[size] = dict()
        for i in range(size):
            twiddle_factors[size][i] = calc_w_pkn(1, i, size)
        size *= 2


def fast_fourier_transform(signal):
    if len(signal) == 2:
        return discrete_fourier_transform(signal)
    half_N = int(len(signal) / 2)
    even_indexes = np.zeros(half_N)
    odd_indexes = np.zeros(half_N)
    for m in range(half_N):
        even_indexes[m] = signal[2 * m]
        odd_indexes[m] = signal[2 * m + 1]

    even_values = fast_fourier_transform(even_indexes)
    odd_values = fast_fourier_transform(odd_indexes)

    result = np.zeros(len(signal), dtype=complex)
    for k in range(half_N):
        twiddle_value = odd_values[k] * twiddle_factors[len(signal)][k]
        result[k] = even_values[k] + twiddle_value
        result[k + half_N] = even_values[k] - twiddle_value
    return result


EXPERIMENTS = 10
fft_times = list()
dft_times = list()
count = list()
signals = list()

for i in range(EXPERIMENTS):
    signal_length = DISCRETE_SAMPLES_NUMBER * (2 ** i)
    signals.append(create_signal(W_MAX, HARMONICS_QUANTITY, signal_length))
    count.append(signal_length)


for i in range(EXPERIMENTS):
    calculate_twiddle_factors(len(signals[i]))
    start_time = time.perf_counter_ns()
    fft = fast_fourier_transform(signals[i])
    deltaTime = time.perf_counter_ns() - start_time
    fft_times.append(deltaTime)
    print(deltaTime)
    if i < 6:
        start_time = time.perf_counter_ns()
        dft = discrete_fourier_transform(signals[i])
        deltaTime = time.perf_counter_ns() - start_time
        dft_times.append(deltaTime)


SIGNAL = create_signal(W_MAX, HARMONICS_QUANTITY, DISCRETE_SAMPLES_NUMBER)

calculate_twiddle_factors(len(SIGNAL))
my_fft = abs(fast_fourier_transform(SIGNAL))
fft = abs(np.fft.fft(SIGNAL))

time_period = DISCRETE_SAMPLES_NUMBER / (2 * W_MAX)


plp.subplot(221)
plp.title('Час виконання FFT від кількості дискретних відліків')
plp.plot(count, fft_times)

plp.subplot(222)
plp.title('Час виконання DFT від кількості дискретних відліків')
plp.plot(count[:6], dft_times)

plp.subplot(223)
plp.title('Швидке перетворення Фур\'є (частотний спектр)')
plp.plot(np.linspace(0, 2 * W_MAX, num=DISCRETE_SAMPLES_NUMBER), my_fft)

plp.subplot(224)
plp.title('Швидке перетворення Фур\'є (з бібліотеки numpy)')
plp.plot(np.linspace(0, 2 * W_MAX, num=DISCRETE_SAMPLES_NUMBER), fft)

plp.show()
