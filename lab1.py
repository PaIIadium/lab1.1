import numpy as np
import matplotlib.pyplot as plp

W_MAX = 2500
HARMONICS_QUANTITY = 14
DISCRETE_SAMPLES_NUMBER = 64


def calculate_value(amplitude, phase, frequency, time):
    return amplitude * np.sin(frequency * time + phase)


def create_signal(w_max, harmonics_quantity, discrete_samples_number):
    sum_values = np.zeros(discrete_samples_number)
    for w in range(1, harmonics_quantity + 1):
        phase = np.random.uniform(-np.pi / 2, np.pi / 2)
        amplitude = np.random.uniform(0.0, 1000.0)
        for t in range(discrete_samples_number):
            sum_values[t] += calculate_value(amplitude, phase, w_max / harmonics_quantity * w, t)
    return sum_values


def create_ccr(signal, copy_signal, max_step):
    auto_correlation_values = np.zeros(max_step)
    math_expectation = np.average(signal)
    std = np.std(signal)
    for i in range(max_step):
        value = 0
        for j in range(len(signal) - i):
            value += signal[j] * copy_signal[j + i]
        offset_math_expectation = np.average(copy_signal[i:])
        offset_std = np.std(copy_signal[i:])
        auto_correlation_values[i] = (value / (len(signal) - i) - math_expectation * offset_math_expectation) / \
                                     (std * offset_std)
    return auto_correlation_values


def create_acr(signal, max_step):
    return create_ccr(signal, signal, max_step)


SIGNAL = create_signal(W_MAX, HARMONICS_QUANTITY, DISCRETE_SAMPLES_NUMBER)
COPY_SIGNAL = create_signal(W_MAX, HARMONICS_QUANTITY, DISCRETE_SAMPLES_NUMBER)

acr = create_acr(SIGNAL, 32)
ccr = create_ccr(SIGNAL, COPY_SIGNAL, 32)

plp.subplot(221)
plp.plot(SIGNAL)
plp.title('Згенерований сигнал')
plp.subplot(222)
plp.plot(COPY_SIGNAL)
plp.title('Копія сигналу')
plp.subplot(223)
plp.plot(acr)
plp.title('Автокореляційна функція')
plp.subplot(224)
plp.plot(ccr)
plp.title('Взаємно кореляційна функція')
plp.show()
