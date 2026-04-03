import re
import sys


def _read_text(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "cp866", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def clean_text(input_file, output_file):
    text = _read_text(input_file)

    # Удалить лишние пробелы
    text = re.sub(r"\s+", " ", text)

    # Удалить HTML теги (если есть)
    text = re.sub(r"<[^>]+>", "", text)

    # Сохранить
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Cleaned: {input_file} → {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_text.py input.txt output.txt")
    else:
        clean_text(sys.argv[1], sys.argv[2])
