import numpy as np
from scipy.io.wavfile import write
import os
import pandas as pd

# 소리 생성 함수 정의
def generate_bee_buzz(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    buzz = 0.5 * np.sin(2 * np.pi * 250 * t)
    modulation = np.sin(2 * np.pi * 5 * t)  # Slow modulation to simulate buzzing
    buzz *= (modulation > 0)  # Create a buzzing effect
    return buzz
def generate_sawing(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    saw = 0.5 * np.sign(np.sin(2 * np.pi * 10 * t)) + 0.3 * np.sin(2 * np.pi * 1000 * t)
    saw *= (np.random.uniform(0, 1, int(sample_rate * duration)) > 0.7)
    return saw

def generate_rain(duration, sample_rate=16000):
    rain = np.random.normal(0, 0.5, int(sample_rate * duration))
    rain = np.convolve(rain, np.ones(1000) / 1000, mode='same')
    return rain

def generate_bell(duration, sample_rate=16000, frequency=440):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    envelope = np.exp(-t * 3)
    bell = envelope * np.sin(2 * np.pi * frequency * t)
    return bell

def generate_thump(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    thump = np.zeros_like(t)
    thump_times = np.random.uniform(0, duration, int(duration * 4))
    for thump_time in thump_times:
        thump += np.exp(-50 * (t - thump_time)) * np.sin(2 * np.pi * 100 * (t - thump_time)) * (t >= thump_time)
    return thump

def generate_wind(duration, sample_rate=16000):
    wind = np.random.normal(0, 0.2, int(sample_rate * duration))
    wind = np.convolve(wind, np.ones(1000) / 1000, mode='same')
    return wind

def generate_waves(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waves = 0.5 * np.sin(2 * np.pi * 0.5 * t) * np.random.normal(0, 0.3, int(sample_rate * duration))
    waves *= (np.random.uniform(0, 1, int(sample_rate * duration)) > 0.8)
    return waves

def generate_footsteps(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    steps = np.zeros_like(t)
    step_times = np.random.uniform(0, duration, int(duration * 2))
    for step_time in step_times:
        steps += np.sin(2 * np.pi * 200 * (t - step_time)) * (t >= step_time)
    steps *= np.exp(-t * 2)
    return steps

def generate_clock_ticking(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    ticking = np.zeros_like(t)
    tick_times = np.arange(0, duration, 1)
    for tick_time in tick_times:
        ticking += np.sin(2 * np.pi * 1000 * (t - tick_time)) * (t >= tick_time)
    ticking *= np.exp(-t * 5)
    return ticking

def generate_machine_noise(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    machine = 0.5 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
    machine *= (np.random.uniform(0, 1, int(sample_rate * duration)) > 0.7)
    return machine

def generate_bird_chirping(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    chirp = np.zeros_like(t)
    chirp_times = np.random.uniform(0, duration, int(duration * 5))
    for chirp_time in chirp_times:
        chirp += 0.5 * np.sin(2 * np.pi * 800 * (t - chirp_time)) * (t >= chirp_time)
    chirp *= np.exp(-5 * t)
    return chirp

def generate_thunder(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    thunder = np.zeros_like(t)
    thunder_times = np.random.uniform(0, duration, int(duration * 2))
    for thunder_time in thunder_times:
        thunder += 0.5 * np.sin(2 * np.pi * 20 * (t - thunder_time)) * np.exp(-3 * (t - thunder_time)) * (
                    t >= thunder_time)
    return thunder

def generate_car_engine(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    engine = np.zeros_like(t)
    engine_times = np.random.uniform(0, duration, int(duration * 3))
    for engine_time in engine_times:
        engine += 0.5 * np.sin(2 * np.pi * 100 * (t - engine_time)) * (t >= engine_time)
    engine += 0.3 * np.sin(2 * np.pi * 200 * t)
    return engine

def generate_dog_barking(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    barking = np.zeros_like(t)
    bark_times = np.random.uniform(0, duration, int(duration * 4))
    for bark_time in bark_times:
        barking += 0.5 * np.sin(2 * np.pi * 500 * (t - bark_time)) * (t >= bark_time)
    barking *= np.exp(-5 * t)
    return barking

def generate_knocking(duration, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    knocking = np.zeros_like(t)
    knock_times = np.random.uniform(0, duration, int(duration * 4))
    for knock_time in knock_times:
        knocking += 0.5 * np.sin(2 * np.pi * 300 * (t - knock_time)) * (t >= knock_time)
    knocking *= np.exp(-5 * t)
    return knocking

# 복합 소리 생성
def generate_combined_sound(duration, sample_rate=16000):
    num_sounds = np.random.randint(3, 6)  # 3에서 5개의 소리 조합
    sounds = []
    sound_types = []
    for _ in range(num_sounds):
        sound_duration = np.random.uniform(1, duration)
        sound, sound_type = generate_random_natural_sound_v3(sound_duration, sample_rate)
        sounds.append(sound[:int(sample_rate * sound_duration)])
        sound_types.append(sound_type)

    max_length = max(len(s) for s in sounds)
    combined = np.zeros(max_length)

    for sound in sounds:
        combined[:len(sound)] += sound

    combined = combined / np.max(np.abs(combined))  # Normalize
    return combined, "_".join(sound_types)

# 소음의 유형을 무작위로 선택하여 사운드 생성
def generate_random_natural_sound_v3(duration, sample_rate=16000):
    sound_type = np.random.choice([
        'bee_buzz', 'sawing', 'rain', 'bell', 'thump',
        'wind', 'waves', 'footsteps', 'clock_ticking', 'machine_noise',
        'bird_chirping', 'thunder', 'car_engine', 'dog_barking', 'knocking'
    ])
    if sound_type == 'bee_buzz':
        return generate_bee_buzz(duration, sample_rate), sound_type
    elif sound_type == 'sawing':
        return generate_sawing(duration, sample_rate), sound_type
    elif sound_type == 'rain':
        return generate_rain(duration, sample_rate), sound_type
    elif sound_type == 'bell':
        return generate_bell(duration, sample_rate, frequency=np.random.uniform(300, 600)), sound_type
    elif sound_type == 'thump':
        return generate_thump(duration, sample_rate), sound_type
    elif sound_type == 'wind':
        return generate_wind(duration, sample_rate), sound_type
    elif sound_type == 'waves':
        return generate_waves(duration, sample_rate), sound_type
    elif sound_type == 'footsteps':
        return generate_footsteps(duration, sample_rate), sound_type
    elif sound_type == 'clock_ticking':
        return generate_clock_ticking(duration, sample_rate), sound_type
    elif sound_type == 'machine_noise':
        return generate_machine_noise(duration, sample_rate), sound_type
    elif sound_type == 'bird_chirping':
        return generate_bird_chirping(duration, sample_rate), sound_type
    elif sound_type == 'thunder':
        return generate_thunder(duration, sample_rate), sound_type
    elif sound_type == 'car_engine':
        return generate_car_engine(duration, sample_rate), sound_type
    elif sound_type == 'dog_barking':
        return generate_dog_barking(duration, sample_rate), sound_type
    elif sound_type == 'knocking':
        return generate_knocking(duration, sample_rate), sound_type

# 사운드 파일 저장
sample_rate = 16000
output_dir = './number6'
os.makedirs(output_dir, exist_ok=True)

sound_generators = [
    generate_bee_buzz,
    generate_sawing,
    generate_rain,
    generate_bell,
    generate_thump,
    generate_wind,
    generate_waves,
    generate_footsteps,
    generate_clock_ticking,
    generate_machine_noise,
    generate_bird_chirping,
    generate_thunder,
    generate_car_engine,
    generate_dog_barking,
    generate_knocking
]

results = []
file_index = 1

# 단일 소리 7000개 생성
while len(results) < 7000:
    generator = np.random.choice(sound_generators)
    duration = np.random.uniform(1, 5)  # 1초에서 5초 사이의 무작위 길이
    sound = generator(duration, sample_rate)
    filename = os.path.join(output_dir, f'single_sound_{file_index:05d}.wav')
    write(filename, sample_rate, sound.astype(np.float32))
    results.append({'path': filename, 'label': 6})
    file_index += 1

# 복합 소리 3000개 생성
while len(results) < 10000:
    duration = np.random.uniform(1, 5)  # 1초에서 5초 사이의 무작위 길이
    sound, combined_type = generate_combined_sound(duration, sample_rate)
    filename = os.path.join(output_dir, f'combined_sound_{file_index:05d}_{combined_type}.wav')
    write(filename, sample_rate, sound.astype(np.float32))
    results.append({'path': filename, 'label': 6})
    file_index += 1

# Create a new dataframe with the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_csv_path = './number6.csv'
results_df.to_csv(output_csv_path, index=False)

print(f"10000개의 다양한 자연음 파일이 생성되고, 경로와 라벨이 {output_csv_path} 파일에 저장되었습니다.")
