import argparse
import math
import shutil
import sys
from os import path
import os
from pathlib import Path

import librosa
import random
import soundfile as sf
import numpy as np
from tqdm import tqdm

#TODO: На будущее: нужно разделить на 3 части: генерация шумов, создание инструкций,
#TODO: генерация аудио по этим инструкциям

# set path to pyAcoustics package, WSJ dataset and root path
pyacoustics_path = './pyAcoustics/'
wsj_path = './LibriMix_WSJ/'
root_path = './'
noise_levels = [3, -2.5, -8.0, -13.5]
subsets = ['tr', 'cv', 'tt']
modes = ['l', 'p']
mix_path = './lombardgrid_2_speakers/wav8k/min/'
continue_writing = False

eps = 1e-12


sys.path.append(path.abspath(pyacoustics_path))
from speech_shaped_noise import generateNoise


def create_folder(path):
    """
    Create a folder if not existent at the specified path.
 
    Args:
        path (str): The path where the folder should be generated.
    """
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def create_noise_folder_structure():
    """
    Creates the output folder structure for the noise files and samples.
 
    Returns:
        output_paths (Dict): The output paths for the noise files.

    """

    output_path_noise_files = root_path + '/data_and_mixing_instructions/Lombardgrid/Noise_Files/'
    create_folder(output_path_noise_files)
    output_paths = {'train' : output_path_noise_files + 'noise_train_val_speakers/',
                'test' : output_path_noise_files + 'noise_test_speakers/'}
    for path in output_paths.values():
        create_folder(path)
    create_folder(output_path_noise_files + 'noise_samples')
    
    return output_paths


def generate_noise_files_per_set():
    """
    Generates noise files for the test and train/validation case separately..
 
    Returns:
        output_paths (Dict): The output paths for the noise files.

    """

    output_paths = create_noise_folder_structure()
    evaluation_dir = wsj_path + 'si_et_05_8k'
    development_dir = wsj_path + 'si_dt_05_8k'
    train_dir = wsj_path + 'si_tr_s_8k_all'
    train_spk_dirs = [os.path.join(train_dir, t) for t in os.listdir(train_dir)]
    test_spk_dirs = [os.path.join(evaluation_dir, t) for t in os.listdir(evaluation_dir)] + [os.path.join(development_dir, t) for t in os.listdir(development_dir)]

    input_paths = {'train': train_spk_dirs,
               'test': test_spk_dirs}
    
    for name, dir_set in input_paths.items():
        print(f"Создается набор шумов {name}...")
        generate_noise_files(dir_set, output_paths[name])

    return output_paths

def max_audio_len(audio_list_path: str | Path, sr: int) -> float:
    """
    Возвращает максимальную длину аудио в секундах.
    Проходит рекурсивно по всем папкам, кроме s1 и s2

    Возвращает 0, если .wav файлов не найдено.
    """

    base = Path(audio_list_path)
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {base}")

    ignore_dirs = {"s1", "s2"}
    max_len = 0.0

    for root, dirs, files in os.walk(base):
        # исключаем каталоги из обхода
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for fname in tqdm(files):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(root, fname)
            try:
                duration = librosa.get_duration(path=fpath, sr=sr)
                if duration > max_len:
                    max_len = duration
            except Exception as e:
                print("Couldn't read %s: %s", fpath, e)

    return max_len

def generate_noise_files(dir_set, output_path):
    """
    Generates for each WSJ0 speaker a speech-shaped noise file.
    
    Args:
        dir_set (list): List of the WSJ0 speaker folders.
        output_path (str): Output path for the noise files.
    
    """

    #Сначала найдем длительность самого долгого файла
    #Длительность шума = max + 1 секунда для надежности
    #sr_path = mix_path + '/l_tr/mix/'
    #clear_sr = librosa.get_samplerate(sr_path + os.listdir(sr_path)[0])
    clear_max_len = 4#math.ceil(max_audio_len(mix_path, clear_sr))
    #Долго считать, так что вычислил заранее, а затем захардкодил 4 секунды
    #Если нужно, можно раскоментарить и проверить правильность

    spk_file_dict = {}
    for dir in dir_set:
        for _,_,files in os.walk(dir):
            for file in files:
                spk_id = os.path.basename(dir)
                if spk_id in spk_file_dict.keys():
                    spk_file_dict[spk_id].append(os.path.join(dir, file))
                else:
                    spk_file_dict[spk_id] = [os.path.join(dir, file)]

    for spk_id in tqdm(spk_file_dict.keys()):
                output_file = output_path + 'ssn_noise_' + str(spk_id) + '.wav'
                if os.path.exists(output_file):
                    print(f"Файл {output_file} уже существует. Пропуск...")
                    continue
                generateNoise(spk_file_dict[spk_id], output_file, clear_max_len + 1)



def rms(x):
    return np.sqrt(np.mean(x.astype(np.float64)**2) + eps)

def scale_noise_to_target_snr(clean, noise, target_snr_db):
    """
    Масштабирует noise так, чтобы SNR(clean, scaled_noise) == target_snr_db (в dB),
    где SNR = 20*log10(rms(clean)/rms(noise)).
    """
    r_clean = rms(clean)
    r_noise = rms(noise)
    desired_r_noise = r_clean / (10**(target_snr_db / 20.0))
    scale = desired_r_noise / (r_noise + eps)
    return noise * scale



def create_mixture_instruction(noise_level, mode, subset, input_paths, delete_clear):
    """
    Creates a two speaker plus noise list for the simulation process and the specified subset. 
    
    Args:
        noise_level (float): Random noise level in dB.
        mode (str): Speech mode (normal or lombard).
        subset (str): Abbreviated name of the subset.
        input_paths (Dict): The input paths for the noise files.

    Returns:
        output_file (str): Path of the generated mixture instruction file. 

    """

    path_noise_output = 'noise_lombardgrid_2_speakers/wav8k/min/' + mode + '_' + subset + '/'
    path_clean = 'lombardgrid_2_speakers/wav8k/min/' + mode + '_' + subset + '/'
    output_path_noise_mixes = root_path + path_noise_output + 'mix/'
    create_folder(output_path_noise_mixes)
    output_file = root_path + 'data_and_mixing_instructions/' + 'mix_l_'+ subset + '_noisy_' + str(noise_level) + '.txt'
    
    if subset == 'tt':
        noise_file_path = input_paths['test']
    else: 
        noise_file_path = input_paths['train']

    instruction_file_wo_noise = open(root_path + '/data_and_mixing_instructions/'+ 'mix_l_'+ subset + '.txt' , 'r')
    noise_file_names = os.listdir(noise_file_path)

    cur_mix_path = mix_path + mode + "_" + subset + '/' + "mix/"

    #Берем частоту дискретизации смеси для определения частоты для шума
    clear_sr = librosa.get_samplerate(cur_mix_path + os.listdir(cur_mix_path)[0])

    noise_files_time_stamps = { noise_filename : 0 for noise_filename in noise_file_names }

    instruction_lines = []

    noise_instruction_path = "./data_and_mixing_instructions/noise_" + mode + "_" + subset + ".txt"
    if os.path.exists(noise_instruction_path) and not continue_writing:  #os.path.getsize(noise_instruction_path) > 0:
        noise_instruction = open(noise_instruction_path,'r')
        noise_read = noise_instruction.read()
        print("Мы в режиме чтения!")
    else:
        noise_instruction = open(noise_instruction_path, 'a+')
        if os.path.getsize(noise_instruction_path) > 0:
            noise_read = noise_instruction.read()
            noise_instruction.seek(0)  # обязательно для a+
            instruction_lines = noise_instruction.read().splitlines()
            noise_instruction.seek(0, os.SEEK_END)
        else:
            noise_read = None


    for s in ['/mix', '/s1', '/s2']:
        os.makedirs(root_path + path_noise_output + s + '/', exist_ok=True)

    for mix_name in tqdm((os.listdir(cur_mix_path))):
        #Проверяем, есть ли такой файл
        if any(mix_name[0:-4] in f for f in instruction_lines):
            print(f"Микс {mix_name[0:-4]} уже существует. Пропуск...")
            continue


        mix = librosa.load(os.path.join(cur_mix_path, mix_name), sr=clear_sr)
        mix_len = len(mix[0])

        if noise_instruction.mode == "a+":
            noise_filename = random.choice(noise_file_names)
        else:
            line_start = noise_read.find(mix_name[0:len(mix_name) - 4])
            line_end = line_start + noise_read[line_start:].find('\n')
            line = noise_read[line_start:line_end]
            noise_filename = line.split(' ')[1] + '.wav'

        noise_file, sr = (librosa.load(os.path.join(noise_file_path, noise_filename), sr=clear_sr))
        if noise_instruction.mode == "a+":
            noise_level = random.choice([3.0, -2.5, -8.0, -13.5])
            target_snr = float(noise_level)
            noise_time = noise_files_time_stamps[noise_filename]
        else:
            #Ищем в файле
            target_snr = float(line.split(' ')[2])
            noise_time = noise_files_time_stamps[noise_filename]


        noise_file_samp = noise_file[
                          noise_time:(noise_time + mix_len)]

        # SNR Шума считается относительно спикера 1
        s1_samp = librosa.load(root_path + path_clean + 's1/' + mix_name)
        scaled_noise = scale_noise_to_target_snr(s1_samp[0], noise_file_samp, (-target_snr))
        noise_mix_samp = scaled_noise + mix[0]
        noise_mix_name = mix_name[0:-4] + '_' + noise_filename.split('_')[2][0:-4] + '_' + f'{target_snr}' + '.wav'

        # предотвратить клиппинг
        peak = np.max(np.abs(noise_mix_samp))
        if peak > 1.0:
            noise_mix_samp = noise_mix_samp / (peak + eps)


        #Create noise mix
        sf.write((output_path_noise_mixes + noise_mix_name), noise_mix_samp, sr)
        if delete_clear:
            os.remove(root_path + path_clean + 'mix/' + mix_name)
        #Copy sources to noise folder
        for s in ['s1/', 's2/']:
            s_clean = root_path + path_clean + s + mix_name
            s_noise = root_path + path_noise_output + s + noise_mix_name
            if os.path.exists(s_clean):
                if delete_clear:
                    shutil.move(s_clean, s_noise)
                else:
                    shutil.copy(s_clean, s_noise)
            else:
                print('Ошибка: файл ' + s_clean + ' не существует')

        #if instruction in write mode it was just created
        #Need to write down instructions
        if noise_instruction.mode == 'a+':
            noise_instruction.write(mix_name[0:len(mix_name)-4] + ' ' + noise_filename[0:len(noise_filename)-4] +
                                    ' ' + str(target_snr) + '\n')
            #noise_files_time_stamps[noise_filename] = noise_files_time_stamps[noise_filename] + mix_len


    noise_instruction.close()
    return noise_instruction

# Это лишнее. Потом удалить.
def duplicate_instruction_files_for_noise_levels(instruction_file, noise_levels, subset):
    """
    Duplicates a two speaker plus noise list for the other specified noise levels and adapts the nosie level. 
    
    Args:
        instruction_file (str): Path to the mixture instruction file that needs to be duplicated.
        noise_levels (list): List of noise levels.
        subset (str): Abbreviated subset name.

    Returns:
        output_file (str): Path of the generated mixture instruction file. 
    
    """
    template = open(instruction_file, 'r')
    template_instructions = template.readlines()

    for level in noise_levels:
        output_file = root_path + 'data_and_mixing_instructions/' + 'mix_l_'+ subset + '_noisy_'+ str(level) + '.txt'
        for m in template_instructions:
            m = m.split()[:-1]
            m = ' '.join(m)
            with open(output_file, 'a') as f:
                f.write(m + ' ' + str(level) + '\n')
     


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-clear", help="Delete clear tracks(sources and mix)",
                        default=False)
    delete_clear = parser.parse_args().delete_clear
    # generate noise files for test set and training/validation set on the basis of the WSJ0 speaker's audio material
    output_paths = generate_noise_files_per_set()
    # for each Lombard subset (training, validation, testing) create random mixing instruction files
    for mode in modes:
        for subset in subsets:
            print(f"Создается набор {mode}_{subset}")
            instruction_file = create_mixture_instruction(noise_levels[0], mode, subset, output_paths, delete_clear)
            #duplicate_instruction_files_for_noise_levels(instruction_file, noise_levels[1:], subset)

    if delete_clear:
        shutil.rmtree("./lombardgrid_2_speakers")
     

if __name__ == "__main__":
    main()
    print("Done")



