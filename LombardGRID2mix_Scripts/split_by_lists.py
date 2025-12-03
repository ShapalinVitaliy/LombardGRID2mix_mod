#!/usr/bin/env python3
"""
split_by_lists.py
Копирует/создаёт ссылки на wav-файлы из исходной папки в структуры:
  <out_root>/Plain/{train,val,test}
  <out_root>/Lombard/{train,val,test}
  <out_root>/Unknown/{...}  (если не удалось определить стиль)
на основе файлов-списков (list files) из репозитория LombardGRID2mix.

Поддерживаемые опции:
  --overwrite       : перезаписывать существующие выходные файлы
"""
import argparse
import logging
import shutil
from tqdm import tqdm
from pathlib import Path
import sys

def find_file_by_token(src_root: Path, token: str):
    token = token.strip()
    if not token:
        return None
    p = Path(token)
    if p.is_absolute() and p.exists():
        return p
    candidate = src_root / token
    if candidate.exists():
        return candidate
    # точный basename / stem совпадение
    for f in src_root.rglob("*.wav"):
        if f.name == token or f.stem == Path(token).stem:
            return f
    # частичный match (включение токена в полный путь)
    for f in src_root.rglob("*.wav"):
        if token in str(f):
            return f
    return None

def detect_condition_from_filename(fname: str):
    ln = fname.lower()
    if "lombard" in ln:
        return "Lombard"
    if "normal" in ln:
        return "Plain"
    return None

def process_list(list_path: Path, src_root: Path, out_root: Path, subset: str,
                 condition: str, overwrite: bool):
    target_base = out_root / condition / subset
    target_base.mkdir(parents=True, exist_ok=True)

    with list_path.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh):

            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            fpath = find_file_by_token(src_root, token)
            if fpath is None:
                logging.warning(f"\nНе найден файл для '{token}' (в списке {list_path})")
                continue
            dest = target_base / fpath.name
            if dest.exists():
                if overwrite:
                    try:
                        if dest.is_symlink() or dest.is_file():
                            dest.unlink()
                    except Exception as e:
                        logging.warning(f"Невозможно удалить существующий {dest}: {e}")


                else:
                    logging.info(f"[SKIP] {dest} уже существует")

            try:
                shutil.copy2(fpath, dest)
            except Exception as e:
                logging.error(f" Ошибка копирования {fpath} -> {dest}: {e}")


def infer_subset_from_name(name: str):
    ln = name.lower()
    if any(x in ln for x in ("tr", "train", "_tr", "-tr")):
        return "tr"
    if any(x in ln for x in ("cv", "val", "dev", "validation", "_cv")):
        return "cv"
    if any(x in ln for x in ("tt", "test", "_tt")):
        return "tt"
    # fallback: положить в train по умолчанию, но выведем предупреждение
    return None

def main():
    parser = argparse.ArgumentParser(description="Split LombardGRID lists into Plain/Lombard train/val/test folders.")
    parser.add_argument("--lists-dir", help="Путь к папке с list-файлами (напр. Lombardgrid/)",
                        default=".\data_and_mixing_instructions\Lombardgrid")
    parser.add_argument("--src-audio", help="Корень с wav-файлами Lombard GRID",
                        default=".\lombardgrid_audio\lombardgrid\\audio")
    parser.add_argument("--out-root", help="Куда поместить структуру <out_root>/{Plain,Lombard,Unknown}/{train,val,test}",
                        default=".\data_and_mixing_instructions\Lombardgrid")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписывать существующие файлы в выходной структуре",
                        default=True)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    lists_dir = Path(args.lists_dir)
    src_root = Path(args.src_audio)
    out_root = Path(args.out_root)
    overwrite = args.overwrite

    if not lists_dir.exists():
        logging.error(f"Папка со списками для копирования файлов не найдена: {lists_dir}")
        sys.exit(2)
    if not src_root.exists():
        logging.error(f"Не найдена папка с источниками: {src_root}")
        sys.exit(2)

    # пройти по всем файлам в lists_dir
    for lf in sorted(lists_dir.rglob("*")):
        if not lf.is_file():
            continue
        # интересуем только текстовые списки (txt/lst или без расширения)
        if lf.suffix.lower() not in ("", ".txt", ".lst", ".csv"):
            continue

        subset = infer_subset_from_name(lf.name)
        if subset is None:
            logging.warning(f"Не удалось определить subset (train/val/test) по имени {lf.name}. Помещаю в 'train' по умолчанию.")
            subset = "train"

        # определение condition
        condition = detect_condition_from_filename(lf.name)
        if condition is None:
            condition = "Unknown"
            logging.warning(f"Не удалось определить стиль для {lf.name}. Помещаю в папку 'Unknown'.")

        logging.info(f"Processing {lf} -> condition={condition}, subset={subset} ...")
        process_list(lf, src_root, out_root, subset, condition, overwrite)

    print("Done.")

if __name__ == "__main__":
    main()