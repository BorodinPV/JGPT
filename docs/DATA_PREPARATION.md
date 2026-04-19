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

### 2.1 Кодировка UTF-8

**Все файлы должны быть в UTF-8.** Это критично для корректной работы токенизатора.

```python
import chardet

def ensure_utf8(filepath):
    """Проверить и конвертировать файл в UTF-8."""
    # Определить теку кодировку
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    detected = chardet.detect(raw)
    encoding = detected['encoding']
    confidence = detected['confidence']
    
    if encoding and encoding.lower() != 'utf-8':
        print(f"  Конвертация {filepath}: {encoding} ({confidence:.0%}) → UTF-8")
        # Прочитать в старой кодировке
        text = raw.decode(encoding, errors='replace')
        # Записать в UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    return False

def convert_directory_to_utf8(directory):
    """Конвертировать все .txt файлы в директории в UTF-8."""
    from pathlib import Path
    
    for txt_file in Path(directory).glob('**/*.txt'):
        try:
            ensure_utf8(txt_file)
        except Exception as e:
            print(f"  ⚠ Ошибка {txt_file}: {e}")
```

**Как проверить кодировку файла:**
```bash
file -i data/books/*.txt        # Linux
chardet data/books/*.txt        # Python chardet
```

### 2.2 Поиск и удаление мусорных символов

Полный список мусорных Unicode символов которые нужно удалить:

```python
def remove_garbage_chars(text):
    """Удалить все мусорные и проблемные символы."""
    
    # 1. Control символы (кроме \n, \t, \r)
    # \x00-\x08: NULL, BACKSPACE, etc
    # \x0b\x0c: VERTICAL TAB, FORM FEED
    # \x0e-\x1f: SHIFT OUT, etc
    # \x7f: DEL
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 2. Unicode replacement character (появляется при ошибках декодирования)
    text = text.replace('\ufffd', '')  # U+FFFD REPLACEMENT CHARACTER
    
    # 3. Unicode BOM и специальные маркеры
    text = text.replace('\ufeff', '')  # U+FEFF BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
    text = text.replace('\ufffe', '')  # U+FFFE (invalid, often appears from bad conversion)
    text = text.replace('\uffff', '')  # U+FFFF (noncharacter)
    
    # 4. Zero-width characters (невидимые, но могут ломать токенизацию)
    text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)  # ZERO WIDTH SPACE, etc
    
    # 5. Soft hyphen и другие невидимые разделители
    text = text.replace('\u00ad', '')  # SOFT HYPHEN
    text = re.sub(r'[\u00a0\u2000-\u200a\u202f\u205f]', ' ', text)  # разные пробелы → обычный
    
    # 6. Directional marks (могут ломать отображение)
    text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)  # LRM, RLM, embedding marks
    
    # 7. Private Use Area (специальные символы шрифтов, не универсальные)
    text = re.sub(r'[\ue000-\uf8ff]', '', text)  # Private Use Area U+E000-U+F8FF
    text = re.sub(r'[\U000f0000-\U000ffffd]', '', text)  # Supplementary Private Use
    
    # 8. Specials и noncharacters
    text = re.sub(r'[\ufff0-\ufffb]', '', text)  # U+FFF0-U+FFFB (noncharacters)
    
    # 9. Variation selectors (не нужны для обучения)
    text = re.sub(r'[\ufe00-\ufe0f]', '', text)  # Variation Selectors-1..16
    
    # 10. Combining characters (диакритика которая может дублироваться)
    # Оставить только если текст на языке где нужна диакритика
    # text = re.sub(r'[\u0300-\u036f]', '', text)  # Combining Diacritical Marks
    
    return text
```

**Расширенный поиск мусора в файлах:**
```python
def scan_for_garbage(filepath):
    """Просканировать файл на наличие мусорных символов."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    garbage_chars = set()
    for i, char in enumerate(text):
        code = ord(char)
        # Control chars (кроме \n, \t, \r)
        if (code < 32 and char not in '\n\t\r') or code == 127:
            garbage_chars.add(f"U+{code:04X} (control)")
        # BOM, replacement, noncharacters
        elif code in (0xFFFD, 0xFFFE, 0xFFFF, 0xFEFF):
            garbage_chars.add(f"U+{code:04X}")
        # Zero-width
        elif 0x200B <= code <= 0x200D or code == 0xFEFF:
            garbage_chars.add(f"U+{code:04X} (zero-width)")
        # Private Use
        elif 0xE000 <= code <= 0xF8FF:
            garbage_chars.add(f"U+{code:04X} (private use)")
    
    if garbage_chars:
        print(f"⚠ {filepath}:")
        for gc in sorted(garbage_chars):
            print(f"   - {gc}")
        return True
    return False
```

### 2.3 Удаление HTML/XML тегов и markup
```python
def remove_artifacts(text):
    # Удалить HTML/XML теги
    text = re.sub(r'<[^>]+>', '', text)
    
    # Удалить Markdown разметку (опционально)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
    
    # Удалить URL
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Удалить email
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    return text
```

### 2.4 Итоговая функция удаления мусора
```python
def clean_garbage(text):
    """Полная очистка от мусорных символов и артефактов."""
    # 1. Удалить control и special chars
    text = remove_garbage_chars(text)
    
    # 2. Удалить HTML/markup
    text = remove_artifacts(text)
    
    # 3. Удалить пустые строки и строки с только пробелами
    lines = text.split('\n')
    lines = [line for line in lines if line.strip()]
    text = '\n'.join(lines)
    
    return text
```

### 2.5 Исправление повторяющейся пунктуации
```python
def fix_repeated_punctuation(text):
    # Заменить ", ," или ". ." на одинарные
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.\s*\.', '…', text)
    text = re.sub(r'!\s*!', '!', text)
    text = re.sub(r'\?\s*\?', '?', text)
    return text
```

## 3. Удаление издательских данных и метаданных

### 3.1 Что удалять

Из книг нужно **обязательно удалять**:

| Тип | Пример | Зачем удалять |
|-----|--------|--------------|
| **Copyright** | `© Иванов И.И., 2020` | Модель запомнит имена/годы |
| **ISBN** | `ISBN 978-5-123456-78-9` | Цифровой мусор |
| **ББК/УДК** | `УДК 821.161.1 ББК 84(2Рос=Рус)` | Библиотечные коды |
| **Издательство** | `Издательство "АСТ", Москва` | Реклама/метаданные |
| **Тираж** | `Тираж 5000 экз.` | Техническая информация |
| **Оглавление** | `Содержание: Глава 1... Глава 2...` | Структура, не текст |
| **Предисловие редактора** | `От издателя: Этот замечательный текст...` | Не основной текст |
| **Послесловие** | `Примечания: Роман написан в...` | Комментарии не автора |
| **Реклама** | `Читайте также: ...` | Реклама других книг |
| **Выходные данные** | `60x90/16. Гарнитура Times. Печать офсетная` | Печатная информация |

### 3.2 Автоматическое удаление паттернов

```python
def remove_publisher_data(text):
    """Удалить издательские данные и метаданные."""
    lines = text.split('\n')
    cleaned_lines = []
    
    skip_next = False
    for line in lines:
        line = line.strip()
        
        # Пропуск пустых строк
        if not line:
            continue
        
        # 1. ISBN
        if re.match(r'ISBN[\s\d\-]+', line, re.IGNORECASE):
            continue
            
        # 2. УДК/ББК
        if re.match(r'(УДК|ББК)\s+[\d\.\(\)]+', line):
            continue
            
        # 3. Copyright
        if re.match(r'©\s+.+\s*,\s*\d{4}', line):
            continue
        if re.match(r'Copyright\s+', line, re.IGNORECASE):
            continue
            
        # 4. Тираж
        if re.match(r'Тираж\s+\d+', line, re.IGNORECASE):
            continue
            
        # 5. Издательство
        if re.match(r'(Издательство|Изд\.?\s+)', line, re.IGNORECASE):
            continue
            
        # 6. Выходные данные (печать)
        if re.match(r'\d+x\d+/\d+', line):  # 60x90/16
            continue
        if re.match(r'(Гарнитура|Формат|Бумага|Печать)', line, re.IGNORECASE):
            continue
            
        # 7. Оглавление/Содержание
        if re.match(r'(Оглавление|Содержание)', line, re.IGNORECASE):
            skip_next = True  # Пропустить следующие строки (оглавление)
            continue
        if skip_next and re.match(r'(Глава|Раздел|\d+\.)', line, re.IGNORECASE):
            continue
        else:
            skip_next = False
            
        # 8. Реклама
        if re.match(r'(Читайте также|Смотрите также|Рекомендуем)', line, re.IGNORECASE):
            continue
            
        # 9. От издателя/редактора
        if re.match(r'(От (издателя|редактора|составителя))', line, re.IGNORECASE):
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)
```

### 3.3 Удаление блоков по маркерам

Иногда метаданные идут блоками в начале/конце файла:

```python
def remove_metadata_blocks(text):
    """Удалить блоки метаданных в начале и конце файла."""
    paragraphs = text.split('\n\n')
    
    if not paragraphs:
        return text
    
    # Удалить первые параграфы если они содержат метаданные
    start_idx = 0
    for i, para in enumerate(paragraphs[:10]):  # Проверить первые 10
        metadata_indicators = [
            'isbn', 'удк', 'ббк', 'copyright', '©', 'тираж', 
            'издательств', 'гарнитура', 'формат', 'печать',
            'оглавление', 'содержание'
        ]
        para_lower = para.lower()
        if any(ind in para_lower for ind in metadata_indicators):
            start_idx = i + 1
        else:
            break
    
    # Удалить последние параграфы если они содержат метаданные
    end_idx = len(paragraphs)
    for i, para in enumerate(reversed(paragraphs[-5:])):  # Проверить последние 5
        metadata_indicators = [
            'isbn', 'удк', 'ббк', 'тираж', 'издательств',
            'читайте также', 'рекомендуем', 'примечания'
        ]
        para_lower = para.lower()
        if any(ind in para_lower for ind in metadata_indicators):
            end_idx = len(paragraphs) - i - 1
        else:
            break
    
    return '\n\n'.join(paragraphs[start_idx:end_idx])
```

## 4. Нормализация текста

### 4.1 Регистр
```python
def normalize_case(text):
    # Привести к нижнему регистру (токенизатор делает это сам)
    return text.lower()
```

### 4.2 Кавычки
```python
def normalize_quotes(text):
    # Заменить ёлочки на обычные (опционально)
    text = text.replace('«', '"').replace('»', '"')
    # Заменить одиночные ёлочки
    text = text.replace('‹', "'").replace('›', "'")
    return text
```

### 4.3 Дефис vs тире
```python
def normalize_dashes(text):
    # Заменить все виды тире на единый вариант
    text = re.sub(r'[—–−]', '—', text)  # em-dash
    # Дефис в словах оставить, тире с пробелами — отдельный знак
    return text
```

## 5. Полный pipeline очистки

```python
def clean_text(text):
    """Полная очистка текста перед обучением."""
    # 1. Удалить control и special символы
    text = remove_garbage_chars(text)
    
    # 2. Удалить HTML/markup
    text = remove_artifacts(text)
    
    # 3. Удалить издательские данные и метаданные
    text = remove_publisher_data(text)
    text = remove_metadata_blocks(text)
    
    # 4. Нормализовать пробелы
    text = normalize_whitespace(text)
    
    # 5. Исправить пунктуацию
    text = fix_punctuation_spacing(text)
    text = fix_repeated_punctuation(text)
    text = fix_ellipsis(text)
    
    # 6. Нормализовать кавычки и тире
    text = normalize_quotes(text)
    text = normalize_dashes(text)
    
    # 7. Финальная нормализация пробелов
    text = normalize_whitespace(text)
    
    return text
```

## 6. Чеклист проверки данных

Перед обучением проверить:

- [ ] **Все файлы в UTF-8** (проверить через `file -i` или `chardet`)
- [ ] Нет BOM символов (U+FEFF)
- [ ] Нет replacement characters (U+FFFD)
- [ ] Нет control символов (кроме \n, \t)
- [ ] Нет zero-width characters
- [ ] Нет private use area символов
- [ ] Нет HTML/XML тегов
- [ ] Нет двойных пробелов
- [ ] Нет пробелов перед знаками пунктуации (кроме тире)
- [ ] Нет повторяющихся знаков пунктуации (,, ,, . . .)
- [ ] Многоточие единое (...) а не . . .
- [ ] Текст в нижнем регистре (токенизатор сделает, но лучше заранее)

## 7. Примеры плохих паттернов

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

## 8. Обработка больших файлов

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

## 9. Валидация после очистки

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

## 10. Рекомендации

1. **Всегда очищайте данные перед обучением** — модель учится на всём что есть в текстах
2. **Конвертируйте в UTF-8** — все файлы должны быть в одной кодировке
3. **Сканируйте на мусор** — запустите `scan_for_garbage()` перед очисткой
4. **Один проход очистки** — лучше чем переучивать модель на плохих данных
5. **Сохраняйте оригиналы** — храните сырые и очищенные тексты отдельно
6. **Проверяйте после очистки** — запустите валидацию на нескольких файлах
7. **Минимальная длина** — убирайте строки короче 10 символов (мусор)
8. **Кодировка** — убедитесь что все файлы в UTF-8 перед токенизацией
