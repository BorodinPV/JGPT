#!/usr/bin/env python3
"""
Download ru-instruct dataset, split into 4 semantic parts,
clean according to DATA_PREPARATION.md rules, and save to data/books/
"""

import os
import re
import sys
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = "data/books"

# ============================================================
# Cleaning functions from DATA_PREPARATION.md
# ============================================================

def remove_garbage_chars(text):
    """Section 2.2: Remove garbage Unicode characters."""
    # Control chars (except \n, \t, \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Replacement character
    text = text.replace('\ufffd', '')
    # BOM and special markers
    text = text.replace('\ufeff', '')
    text = text.replace('\ufffe', '')
    text = text.replace('\uffff', '')
    # Zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
    # Soft hyphen and special spaces
    text = text.replace('\u00ad', '')
    text = re.sub(r'[\u00a0\u2000-\u200a\u202f\u205f]', ' ', text)
    # Directional marks
    text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)
    # Private Use Area
    text = re.sub(r'[\ue000-\uf8ff]', '', text)
    text = re.sub(r'[\U000f0000-\U000ffffd]', '', text)
    # Noncharacters
    text = re.sub(r'[\ufff0-\ufffb]', '', text)
    # Variation selectors
    text = re.sub(r'[\ufe00-\ufe0f]', '', text)
    return text


def remove_artifacts(text):
    """Section 2.3: Remove HTML/XML tags, markup, URLs, emails."""
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove Markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    return text


def normalize_whitespace(text):
    """Section 1.1: Normalize whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def fix_punctuation_spacing(text):
    """Section 1.2: Fix spaces around punctuation."""
    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.!?;:"])', r'\1', text)
    # Add space after punctuation (except dot in numbers)
    text = re.sub(r'([,.!?;:])(?=[^\s\d])', r'\1 ', text)
    # Remove double spaces after punctuation
    text = re.sub(r'[,.!?;:]\s{2,}', lambda m: m.group(0)[0] + ' ', text)
    return text


def fix_ellipsis(text):
    """Section 1.3: Unified ellipsis."""
    text = re.sub(r'\.\s*\.\s*\.', '...', text)
    return text


def fix_repeated_punctuation(text):
    """Section 2.5: Fix repeated punctuation."""
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'!\s*!', '!', text)
    text = re.sub(r'\?\s*\?', '?', text)
    return text


def normalize_quotes(text):
    """Section 4.2: Normalize quotes."""
    text = text.replace('\u00ab', '"').replace('\u00bb', '"')
    text = text.replace('\u2039', "'").replace('\u203a', "'")
    return text


def normalize_dashes(text):
    """Section 4.3: Normalize dashes."""
    text = re.sub(r'[\u2014\u2013\u2212]', '\u2014', text)
    return text


def remove_publisher_data(text):
    """Section 3.2: Remove publisher data and metadata patterns."""
    lines = text.split('\n')
    cleaned_lines = []

    skip_next = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # ISBN
        if re.match(r'ISBN[\s\d\-]+', line, re.IGNORECASE):
            continue
        # УДК/ББК
        if re.match(r'(УДК|ББК)\s+[\d\.\(\)]+', line):
            continue
        # Copyright
        if re.match(r'©\s+.+\s*,\s*\d{4}', line):
            continue
        if re.match(r'Copyright\s+', line, re.IGNORECASE):
            continue
        # Тираж
        if re.match(r'Тираж\s+\d+', line, re.IGNORECASE):
            continue
        # Издательство
        if re.match(r'(Издательство|Изд\.?\s+)', line, re.IGNORECASE):
            continue
        # Выходные данные
        if re.match(r'\d+x\d+/\d+', line):
            continue
        if re.match(r'(Гарнитура|Формат|Бумага|Печать)', line, re.IGNORECASE):
            continue
        # Оглавление
        if re.match(r'(Оглавление|Содержание)', line, re.IGNORECASE):
            skip_next = True
            continue
        if skip_next and re.match(r'(Глава|Раздел|\d+\.)', line, re.IGNORECASE):
            continue
        else:
            skip_next = False
        # Реклама
        if re.match(r'(Читайте также|Смотрите также|Рекомендуем)', line, re.IGNORECASE):
            continue
        # От издателя
        if re.match(r'(От (издателя|редактора|составителя))', line, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_metadata_blocks(text):
    """Section 3.3: Remove metadata blocks at start/end."""
    paragraphs = text.split('\n\n')
    if not paragraphs:
        return text

    start_idx = 0
    for i, para in enumerate(paragraphs[:10]):
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

    end_idx = len(paragraphs)
    for i, para in enumerate(reversed(paragraphs[-5:])):
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


def clean_text(text):
    """Section 5: Full cleaning pipeline."""
    # 1. Remove control and special chars
    text = remove_garbage_chars(text)
    # 2. Remove HTML/markup
    text = remove_artifacts(text)
    # 3. Remove publisher data
    text = remove_publisher_data(text)
    text = remove_metadata_blocks(text)
    # 4. Normalize whitespace
    text = normalize_whitespace(text)
    # 5. Fix punctuation
    text = fix_punctuation_spacing(text)
    text = fix_repeated_punctuation(text)
    text = fix_ellipsis(text)
    # 6. Normalize quotes and dashes
    text = normalize_quotes(text)
    text = normalize_dashes(text)
    # 7. Lowercase
    text = text.lower()
    # 8. Final whitespace normalization
    text = normalize_whitespace(text)
    return text


# ============================================================
# Dataset conversion: conversations -> text format
# ============================================================

def conversation_to_text(conversations):
    """
    Convert a conversation list to a single text string.
    Format: concatenates all turns as continuous text.
    Each turn is separated by newline.
    Keeps Q+A together as one unit (never splits them).
    """
    parts = []
    for turn in conversations:
        role = turn.get('role', '')
        content = turn.get('content', '')
        if content and content.strip():
            parts.append(content.strip())
    return '\n'.join(parts)


# ============================================================
# Semantic split into 4 parts
# ============================================================

def get_semantic_group(source):
    """Assign a semantic group based on source dataset."""
    if source in ('OpenOrca-ru', 'OpenHermes-2.5-ru', 'dolphin-ru'):
        return 'general'  # General knowledge & reasoning
    elif source == 'alpaca-cleaned-ru':
        return 'instructions'  # Instruction following
    elif source == 'conala-mined-ru':
        return 'code'  # Code snippets
    elif source in ('gsm8k-ru', 'boolq-ru'):
        return 'math_qa'  # Math + QA
    else:
        return 'general'  # Fallback


GROUP_NAMES = {
    'general': 'ru_instruct_general',
    'instructions': 'ru_instruct_instructions',
    'code': 'ru_instruct_code',
    'math_qa': 'ru_instruct_math_qa',
}

# Maximum file size in bytes (~200 MB for safe Java processing)
MAX_FILE_SIZE = 200 * 1024 * 1024


def split_into_chunks(texts, max_size):
    """
    Split texts into chunks without breaking individual texts.
    Each chunk will be at most max_size bytes when written.
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for text in texts:
        # Size of this text with separators
        text_size = len(text.encode('utf-8')) + 4  # +4 for '\n\n'
        
        # If adding this text exceeds max_size, save current chunk
        if current_chunk and current_size + text_size > max_size:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        
        current_chunk.append(text)
        current_size += text_size
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def split_and_save():
    """Download dataset, split by meaning, clean, and save."""
    print("Loading dataset d0rj/ru-instruct...")
    ds = load_dataset('d0rj/ru-instruct')
    train = ds['train']
    print(f"Total rows: {len(train)}")

    # Group data
    groups = {g: [] for g in GROUP_NAMES}
    for i, item in enumerate(train):
        source = item['source']
        group = get_semantic_group(source)
        text = conversation_to_text(item['conversations'])
        groups[group].append(text)

        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1}/{len(train)}...")

    # Print stats
    print("\nDataset split:")
    for group, texts in groups.items():
        total_chars = sum(len(t) for t in texts)
        print(f"  {GROUP_NAMES[group]}: {len(texts)} items, {total_chars:,} chars")

    # Clean and save each group
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for group, texts in groups.items():
        if not texts:
            print(f"\nSkipping {GROUP_NAMES[group]} (empty)")
            continue

        print(f"\nCleaning {GROUP_NAMES[group]}...")

        # Clean all texts
        cleaned_texts = []
        for j, text in enumerate(texts):
            cleaned = clean_text(text)
            if cleaned and len(cleaned) > 10:  # Skip very short texts
                cleaned_texts.append(cleaned)

            if (j + 1) % 50000 == 0:
                print(f"  Cleaned {j + 1}/{len(texts)}...")

        # Split into chunks if needed
        chunks = split_into_chunks(cleaned_texts, MAX_FILE_SIZE)
        total_chars = sum(len(t) for t in cleaned_texts)
        
        if len(chunks) == 1:
            # No split needed
            filename = GROUP_NAMES[group] + '.txt'
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                for text in cleaned_texts:
                    f.write(text + '\n\n')
            print(f"  Saved {filepath}: {len(cleaned_texts)} texts, {total_chars:,} chars")
        else:
            # Save multiple chunks
            for idx, chunk in enumerate(chunks, 1):
                filename = f"{GROUP_NAMES[group]}_part{idx}.txt"
                filepath = os.path.join(OUTPUT_DIR, filename)
                chunk_chars = sum(len(t) for t in chunk)
                with open(filepath, 'w', encoding='utf-8') as f:
                    for text in chunk:
                        f.write(text + '\n\n')
                print(f"  Saved {filepath}: {len(chunk)} texts, {chunk_chars:,} chars")

    print("\nDone!")


if __name__ == '__main__':
    split_and_save()
