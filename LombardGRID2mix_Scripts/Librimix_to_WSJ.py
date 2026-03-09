import os

import random
from pathlib import Path
#Отформатировать из flac в wav
import soundfile as sf
from tqdm import tqdm


#data, samplerate = sf.read("input.flac")
#sf.write("output.wav", data, samplerate)

SEED = 1234

def main():
    librimix_path = Path("./train-clean-360")
    output_path = Path("./LibriMix_WSJ/")
    train_dir = "si_tr_s_8k_all"
    eval_dir = "si_et_05_8k"
    dev_dir = "si_dt_05_8k"

    spkr_list = []
    out_dict = {}

    random.seed(SEED)

    for spkr in librimix_path.iterdir():
        out_dict[spkr] = []
        for file in spkr.rglob("*.flac"):
            out_dict[spkr].append(file)

    keys = list(out_dict.keys())
    random.shuffle(keys)

    n = len(keys)
    i1 = int(n * 0.8)
    i2 = int(n * (0.8 + 0.1))
    # защита от выхода за границы
    i1 = max(0, min(i1, n))
    i2 = max(i1, min(i2, n))
    keys_train = keys[:i1]
    keys_val = keys[i1:i2]
    keys_dev = keys[i2:]

    train = {k: out_dict[k] for k in keys_train}
    eval = {k: out_dict[k] for k in keys_val}
    dev = {k: out_dict[k] for k in keys_dev}

    out_dict = {train_dir:train, eval_dir:eval, dev_dir:dev}



    for subset, subset_list in out_dict.items():
        subset_path = os.path.join(output_path, subset)
        print(f"Создаем {subset_path}")
        for spkr in tqdm(subset_list):
            spkr_path = os.path.join(subset_path, spkr.name)
            os.makedirs(spkr_path, exist_ok=True)
            for file in subset_list[spkr]:
                file_path = os.path.join(spkr_path, f"{file.stem}.wav")
                data, samplerate = sf.read(file)
                sf.write(file_path, data, samplerate)

if __name__ == "__main__":
    main()