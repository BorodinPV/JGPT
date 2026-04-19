# Правила подготовки текстовых данных для обучения JGPT

## 1. Очистка пробелов

### 1.1 Нормализация пробелов
```python
import re

def normalize_whitespace(text):
    # Заменить все виды пробелов (табы, неразрывные, etc) на обычные
    text = re.sub(r'\s+', ' ', text)
    # Убрать пробелы в начале и конце
    text = text.strip()
    return text
```

### 1.2 Пробелы вокруг пунктуации
```python
def fix_punctuation_spacing(text):
    # Убрать пробелы перед знаками пунктуации
    text = re.sub(r'\s+([,.!?;:"])', r'\1', text)
    # Добавить пробел после знаков пунктуации (кроме точки в конце числа)
    text = re.sub(r'([,.!?;:])(?=[^\s\d])', r'\1 ', text)
    # Убрать двойные пробелы после пунктуации
    text = re.sub(r'[,.!?;:]\s{2,}', lambda m: m.group(0)[0] + ' ', text)
    return text
```

### 1.3 Многоточие
```python
def fix_ellipsis(text):
    # Заменить "..." или ". . ." на единое многоточие
    text = re.sub(r'\.\s*\.\s*\.', '…', text)
    # Или заменить на три точки без пробелов
    text = re.sub(r'\.\s*\.\s*\.', '...', text)
    return text
```

## 2. Очистка артефактов

### 2.1 Удаление мусора
```python
def remove_artifacts(text):
    # Удалить/control символы (кроме \n, \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Удалить Unicode мусор
    text = re.sub(r'[\ufffd\ufffe\uffff\ufeff]', '', text)
    # Удалить HTML/XML теги
    text = re.sub(r'<[^>]+>', '', text)
    return text
```

### 2.2 Исправление повторяющейся пунктуации
```python
def fix_repeated_punctuation(text):
    # Заменить ", ," или ". ." на одинарные
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.\s*\.', '…', text)
    text = re.sub(r'!\s*!', '!', text)
    text = re.sub(r'\?\s*\?', '?', text)
    return text
```

## 3. Нормализация текста

### 3.1 Регистр
```python
def normalize_case(text):
    # Привести к нижнему регистру (токенизатор делает это сам)
    return text.lower()
```

### 3.2 Кавычки
```python
def normalize_quotes(text):
    # Заменить ёлочки на обычные (опционально)
    text = text.replace('«', '"').replace('»', '"')
    # Заменить одиночные ёлочки
    text = text.replace('‹', "'").replace('›', "'")
    return text
```

### 3.3 Дефис vs тире
```python
def normalize_dashes(text):
    # Заменить все виды тире на единый вариант
    text = re.sub(r'[—–−]', '—', text)  # em-dash
    # Дефис в словах оставить, тире с пробелами — отдельный знак
    return text
```

## 4. Полный pipeline очистки

```python
def clean_text(text):
    """Полная очистка текста перед обучением."""
    # 1. Удалить артефакты
    text = remove_artifacts(text)
    
    # 2. Нормализовать пробелы
    text = normalize_whitespace(text)
    
    # 3. Исправить пунктуацию
    text = fix_punctuation_spacing(text)
    text = fix_repeated_punctuation(text)
    text = fix_ellipsis(text)
    
    # 4. Нормализовать кавычки и тире
    text = normalize_quotes(text)
    text = normalize_dashes(text)
    
    # 5. Финальная нормализация пробелов
    text = normalize_whitespace(text)
    
    return text
```

## 5. Чеклист проверки данных

Перед обучением проверить:

- [ ] Нет двойных пробелов
- [ ] Нет пробелов перед знаками пунктуации (кроме тире)
- [ ] Нет повторяющихся знаков пунктуации (,, ,, . . .)
- [ ] Многоточие единое (...) а не . . .
- [ ] Нет control символов
- [ ] Нет HTML тегов
- [ ] Текст в нижнем регистре (токенизатор сделает, но лучше заранее)

## 6. Примеры плохих паттернов

### ❌ Плохо:
```
поход от варева на суккур , и как же я выйду замуж . . .
текст  с   двойными    пробелами
что - то , я не знаю
```

### ✅ Хорошо:
```
поход от варева на суккур, и как же я выйду замуж...
текст с двойными пробелами
что-то, я не знаю
```

## 7. Обработка больших файлов

```python
import os
from pathlib import Path

def clean_book_file(input_path, output_path):
    """Очистить один файл книги."""
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    cleaned = clean_text(text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"✓ {input_path} → {output_path} ({len(text)} → {len(cleaned)} chars)")

def clean_all_books(data_dir='data/books'):
    """Очистить все книги в директории."""
    input_dir = Path(data_dir)
    output_dir = Path(data_dir + '_cleaned')
    output_dir.mkdir(exist_ok=True)
    
    for txt_file in input_dir.glob('**/*.txt'):
        output_file = output_dir / txt_file.relative_to(input_dir)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        clean_book_file(txt_file, output_file)
    
    print(f"\nГотово! Очищенные файлы в {output_dir}")

if __name__ == '__main__':
    clean_all_books()
```

## 8. Валидация после очистки

```python
def validate_cleaned_text(text, filename='unknown'):
    """Проверить текст на типичные проблемы."""
    issues = []
    
    if re.search(r'\s{2,}', text):
        issues.append('двойные пробелы')
    
    if re.search(r'\s+[,.!?;:"]', text):
        issues.append('пробелы перед пунктуацией')
    
    if re.search(r'[,.!?;:"]\s{2,}', text):
        issues.append('множественные пробелы после пунктуации')
    
    if re.search(r'[,.!?;:"]\s+[,.!?;:"]', text):
        issues.append('повторяющаяся пунктуация')
    
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text):
        issues.append('control символы')
    
    if issues:
        print(f"⚠ {filename}: {', '.join(issues)}")
        return False
    else:
        print(f"✓ {filename}: OK")
        return True
```

## 9. Рекомендации

1. **Всегда очищайте данные перед обучением** — модель учится на всём что есть в текстах
2. **Один проход очистки** — лучше чем переучивать модель на плохих данных
3. **Сохраняйте оригиналы** — храните сырые и очищенные тексты отдельно
4. **Проверяйте после очистки** — запустите валидацию на нескольких файлах
5. **Минимальная длина** — убирайте строки короче 10 символов (мусор)
6. **Кодировка** — убедитесь что все файлы в UTF-8
