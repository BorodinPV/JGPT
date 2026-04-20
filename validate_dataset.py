#!/usr/bin/env python3
"""
Validation script for dataset files based on DATA_PREPARATION.md rules.
Checks for all the issues described in the data preparation guidelines.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# ============================================================
# Checks from DATA_PREPARATION.md
# ============================================================

def check_utf8_bom(filepath):
    """Check for BOM bytes at start of file."""
    with open(filepath, 'rb') as f:
        first_bytes = f.read(3)
    return first_bytes[:3] == b'\xef\xbb\xbf'


def scan_for_garbage_chars(text, filepath):
    """Scan for garbage Unicode characters (Section 2.2)."""
    garbage = defaultdict(int)
    for i, char in enumerate(text):
        code = ord(char)
        # Control chars (except \n, \t, \r)
        if (code < 32 and char not in '\n\t\r') or code == 127:
            garbage[f"U+{code:04X} (control)"] += 1
        # Replacement character
        elif code == 0xFFFD:
            garbage["U+FFFD (replacement)"] += 1
        # BOM / zero-width
        elif code == 0xFEFF:
            garbage["U+FEFF (BOM/zero-width)"] += 1
        elif code == 0xFFFE:
            garbage["U+FFFE (invalid)"] += 1
        elif code == 0xFFFF:
            garbage["U+FFFF (noncharacter)"] += 1
        # Zero-width
        elif code in (0x200B, 0x200C, 0x200D, 0x2060):
            garbage[f"U+{code:04X} (zero-width)"] += 1
        # Soft hyphen
        elif code == 0x00AD:
            garbage["U+00AD (soft hyphen)"] += 1
        # Directional marks
        elif code in (0x200E, 0x200F) or (0x202A <= code <= 0x202E):
            garbage[f"U+{code:04X} (directional)"] += 1
        # Private Use Area
        elif 0xE000 <= code <= 0xF8FF:
            garbage[f"U+{code:04X} (private use)"] += 1
        # Variation selectors
        elif 0xFE00 <= code <= 0xFE0F:
            garbage[f"U+{code:04X} (variation selector)"] += 1

    return dict(garbage)


def check_double_spaces(text):
    """Check for double/multiple spaces (Section 6 checklist)."""
    matches = re.findall(r' {2,}', text)
    return len(matches)


def check_space_before_punctuation(text):
    """Check for spaces before punctuation (Section 6 checklist)."""
    # Spaces before , . ! ? ; : (but not before em-dash)
    matches = re.findall(r'\s+[,.;:!?"]', text)
    return len(matches)


def check_repeated_punctuation(text):
    """Check for repeated punctuation like ,, . . (Section 6 checklist)."""
    issues = 0
    issues += len(re.findall(r',\s*,', text))
    issues += len(re.findall(r'\.\s*\.', text))
    issues += len(re.findall(r'!\s*!', text))
    issues += len(re.findall(r'\?\s*\?', text))
    return issues


def check_html_tags(text):
    """Check for HTML/XML tags (Section 2.3)."""
    return len(re.findall(r'<[^>]+>', text))


def check_isbn(text):
    """Check for ISBN patterns (Section 3.1)."""
    return len(re.findall(r'ISBN[\s\d\-]+', text, re.IGNORECASE))


def check_udk_bbk(text):
    """Check for УДК/ББК patterns (Section 3.1)."""
    return len(re.findall(r'(УДК|ББК)\s+[\d\.\(\)]+', text))


def check_copyright(text):
    """Check for copyright patterns (Section 3.1)."""
    issues = 0
    issues += len(re.findall(r'©\s+.+\s*,\s*\d{4}', text))
    issues += len(re.findall(r'Copyright\s+', text, re.IGNORECASE))
    return issues


def check_tirazh(text):
    """Check for print run info (Section 3.1)."""
    return len(re.findall(r'Тираж\s+\d+', text, re.IGNORECASE))


def check_oglavlenie(text):
    """Check for table of contents (Section 3.1)."""
    return len(re.findall(r'(Оглавление|Содержание)', text, re.IGNORECASE))


def check_urls(text):
    """Check for URLs (Section 2.3)."""
    return len(re.findall(r'https?://\S+', text))


def check_emails(text):
    """Check for email addresses (Section 2.3)."""
    return len(re.findall(r'\S+@\S+\.\S+', text))


def check_short_lines(text, min_len=10):
    """Check for lines shorter than min_len chars (Section 10, recommendation 7)."""
    lines = text.split('\n')
    short = [l for l in lines if l.strip() and len(l.strip()) < min_len]
    return len(short)


def check_uppercase_ratio(text):
    """Check if text has been lowercased (Section 4.1, Section 6 checklist)."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    upper = sum(1 for c in alpha_chars if c.isupper())
    return upper / len(alpha_chars)


def check_ellipsis_scattered(text):
    """Check for '. . .' pattern instead of unified ellipsis (Section 1.3)."""
    return len(re.findall(r'\.\s*\.\s*\.', text))


def check_markdown_markup(text):
    """Check for remaining markdown markup (Section 2.3)."""
    issues = 0
    issues += len(re.findall(r'\*\*[^*]+\*\*', text))  # bold
    issues += len(re.findall(r'(?<!\*)\*[^*]+\*(?!\*)', text))  # italic
    issues += len(re.findall(r'__[^_]+__', text))  # bold
    return issues


# ============================================================
# Main validation
# ============================================================

def validate_file(filepath):
    """Run all checks on a single file."""
    results = {}

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception as e:
        return {'error': str(e)}

    results['size'] = len(text)
    results['has_bom'] = check_utf8_bom(filepath)
    results['garbage_chars'] = scan_for_garbage_chars(text, filepath)
    results['double_spaces'] = check_double_spaces(text)
    results['space_before_punct'] = check_space_before_punctuation(text)
    results['repeated_punct'] = check_repeated_punctuation(text)
    results['html_tags'] = check_html_tags(text)
    results['isbn'] = check_isbn(text)
    results['udk_bbk'] = check_udk_bbk(text)
    results['copyright'] = check_copyright(text)
    results['tirazh'] = check_tirazh(text)
    results['oglavlenie'] = check_oglavlenie(text)
    results['urls'] = check_urls(text)
    results['emails'] = check_emails(text)
    results['short_lines'] = check_short_lines(text)
    results['uppercase_ratio'] = check_uppercase_ratio(text)
    results['scattered_ellipsis'] = check_ellipsis_scattered(text)
    results['markdown_markup'] = check_markdown_markup(text)

    return results


def summarize_results(all_results, data_dirs):
    """Print a summary of all validation results."""
    total_files = len(all_results)
    total_chars = sum(r.get('size', 0) for r in all_results.values())

    print("=" * 80)
    print(f"VALIDATION REPORT")
    print(f"Directories: {', '.join(data_dirs)}")
    print(f"Total files: {total_files}")
    print(f"Total characters: {total_chars:,}")
    print("=" * 80)

    # Aggregate counters
    counters = {
        'has_bom': 0,
        'garbage_files': 0,
        'double_spaces': 0,
        'space_before_punct': 0,
        'repeated_punct': 0,
        'html_tags': 0,
        'isbn': 0,
        'udk_bbk': 0,
        'copyright': 0,
        'tirazh': 0,
        'oglavlenie': 0,
        'urls': 0,
        'emails': 0,
        'short_lines': 0,
        'not_lowercased': 0,
        'scattered_ellipsis': 0,
        'markdown_markup': 0,
    }

    total_garbage_types = defaultdict(int)
    total_double_spaces = 0
    total_space_before_punct = 0
    total_repeated_punct = 0
    total_html_tags = 0
    total_isbn = 0
    total_udk_bbk = 0
    total_copyright = 0
    total_tirazh = 0
    total_oglavlenie = 0
    total_urls = 0
    total_emails = 0
    total_short_lines = 0
    total_scattered_ellipsis = 0
    total_markdown = 0

    avg_uppercase_ratios = []

    for filepath, r in all_results.items():
        if 'error' in r:
            continue

        if r['has_bom']:
            counters['has_bom'] += 1

        if r['garbage_chars']:
            counters['garbage_files'] += 1
            for gc, count in r['garbage_chars'].items():
                total_garbage_types[gc] += count

        if r['double_spaces'] > 0:
            counters['double_spaces'] += 1
            total_double_spaces += r['double_spaces']

        if r['space_before_punct'] > 0:
            counters['space_before_punct'] += 1
            total_space_before_punct += r['space_before_punct']

        if r['repeated_punct'] > 0:
            counters['repeated_punct'] += 1
            total_repeated_punct += r['repeated_punct']

        if r['html_tags'] > 0:
            counters['html_tags'] += 1
            total_html_tags += r['html_tags']

        if r['isbn'] > 0:
            counters['isbn'] += 1
            total_isbn += r['isbn']

        if r['udk_bbk'] > 0:
            counters['udk_bbk'] += 1
            total_udk_bbk += r['udk_bbk']

        if r['copyright'] > 0:
            counters['copyright'] += 1
            total_copyright += r['copyright']

        if r['tirazh'] > 0:
            counters['tirazh'] += 1
            total_tirazh += r['tirazh']

        if r['oglavlenie'] > 0:
            counters['oglavlenie'] += 1
            total_oglavlenie += r['oglavlenie']

        if r['urls'] > 0:
            counters['urls'] += 1
            total_urls += r['urls']

        if r['emails'] > 0:
            counters['emails'] += 1
            total_emails += r['emails']

        if r['short_lines'] > 0:
            counters['short_lines'] += 1
            total_short_lines += r['short_lines']

        if r['uppercase_ratio'] > 0.01:  # more than 1% uppercase
            counters['not_lowercased'] += 1
            avg_uppercase_ratios.append(r['uppercase_ratio'])

        if r['scattered_ellipsis'] > 0:
            counters['scattered_ellipsis'] += 1
            total_scattered_ellipsis += r['scattered_ellipsis']

        if r['markdown_markup'] > 0:
            counters['markdown_markup'] += 1
            total_markdown += r['markdown_markup']

    print()
    print("ISSUES FOUND:")
    print("-" * 80)

    # BOM
    print(f"  {'✗' if counters['has_bom'] else '✓'} BOM symbols:          {counters['has_bom']} files")

    # Garbage chars
    print(f"  {'✗' if counters['garbage_files'] else '✓'} Files with garbage:  {counters['garbage_files']} files")
    if total_garbage_types:
        for gc, count in sorted(total_garbage_types.items(), key=lambda x: -x[1]):
            print(f"      {gc}: {count:,} occurrences")

    # Double spaces
    status = '✗' if counters['double_spaces'] else '✓'
    print(f"  {status} Double spaces:         {counters['double_spaces']} files ({total_double_spaces:,} occurrences)")

    # Space before punctuation
    status = '✗' if counters['space_before_punct'] else '✓'
    print(f"  {status} Space before punct:    {counters['space_before_punct']} files ({total_space_before_punct:,} occurrences)")

    # Repeated punctuation
    status = '✗' if counters['repeated_punct'] else '✓'
    print(f"  {status} Repeated punctuation:  {counters['repeated_punct']} files ({total_repeated_punct:,} occurrences)")

    # HTML tags
    status = '✗' if counters['html_tags'] else '✓'
    print(f"  {status} HTML tags:            {counters['html_tags']} files ({total_html_tags:,} occurrences)")

    # ISBN
    status = '✗' if counters['isbn'] else '✓'
    print(f"  {status} ISBN patterns:        {counters['isbn']} files ({total_isbn} occurrences)")

    # УДК/ББК
    status = '✗' if counters['udk_bbk'] else '✓'
    print(f"  {status} УДК/ББК patterns:     {counters['udk_bbk']} files ({total_udk_bbk} occurrences)")

    # Copyright
    status = '✗' if counters['copyright'] else '✓'
    print(f"  {status} Copyright:            {counters['copyright']} files ({total_copyright} occurrences)")

    # Тираж
    status = '✗' if counters['tirazh'] else '✓'
    print(f"  {status} Тираж:                {counters['tirazh']} files ({total_tirazh} occurrences)")

    # Оглавление
    status = '✗' if counters['oglavlenie'] else '✓'
    print(f"  {status} Оглавление:           {counters['oglavlenie']} files ({total_oglavlenie} occurrences)")

    # URLs
    status = '✗' if counters['urls'] else '✓'
    print(f"  {status} URLs:                 {counters['urls']} files ({total_urls} occurrences)")

    # Emails
    status = '✗' if counters['emails'] else '✓'
    print(f"  {status} Emails:               {counters['emails']} files ({total_emails} occurrences)")

    # Short lines
    status = '✗' if counters['short_lines'] else '✓'
    print(f"  {status} Short lines (<10ch):   {counters['short_lines']} files ({total_short_lines:,} lines)")

    # Not lowercased
    status = '✗' if counters['not_lowercased'] else '✓'
    avg_ratio = sum(avg_uppercase_ratios) / len(avg_uppercase_ratios) if avg_uppercase_ratios else 0
    print(f"  {status} Not lowercased:        {counters['not_lowercased']} files (avg uppercase ratio: {avg_ratio:.1%})")

    # Scattered ellipsis
    status = '✗' if counters['scattered_ellipsis'] else '✓'
    print(f"  {status} Scattered ellipsis:    {counters['scattered_ellipsis']} files ({total_scattered_ellipsis:,} occurrences)")

    # Markdown
    status = '✗' if counters['markdown_markup'] else '✓'
    print(f"  {status} Markdown markup:       {counters['markdown_markup']} files ({total_markdown:,} occurrences)")

    print()
    print("=" * 80)

    # Show worst offenders
    print("TOP 10 WORST FILES (by total issues):")
    print("-" * 80)

    file_scores = {}
    for filepath, r in all_results.items():
        if 'error' in r:
            continue
        score = (
            r['double_spaces'] +
            r['space_before_punct'] +
            r['repeated_punct'] +
            r['html_tags'] +
            r['isbn'] +
            r['udk_bbk'] +
            r['copyright'] +
            r['tirazh'] +
            r['oglavlenie'] +
            r['urls'] +
            r['emails'] +
            r['short_lines'] +
            r['scattered_ellipsis'] +
            r['markdown_markup']
        )
        if score > 0:
            file_scores[filepath] = score

    sorted_files = sorted(file_scores.items(), key=lambda x: -x[1])[:10]
    for filepath, score in sorted_files:
        r = all_results[filepath]
        print(f"  {score:>8,} issues  {filepath}")
        details = []
        if r['double_spaces']: details.append(f"double_spaces={r['double_spaces']}")
        if r['space_before_punct']: details.append(f"space_before_punct={r['space_before_punct']}")
        if r['repeated_punct']: details.append(f"repeated_punct={r['repeated_punct']}")
        if r['html_tags']: details.append(f"html_tags={r['html_tags']}")
        if r['short_lines']: details.append(f"short_lines={r['short_lines']}")
        if r['scattered_ellipsis']: details.append(f"ellipsis={r['scattered_ellipsis']}")
        if r['garbage_chars']: details.append(f"garbage={sum(r['garbage_chars'].values())}")
        if details:
            print(f"            {', '.join(details[:5])}")

    print()


def show_sample_issues(all_results, max_samples=3):
    """Show actual text samples of issues found."""
    print("SAMPLE ISSUES:")
    print("-" * 80)

    # Find files with space before punctuation
    for filepath, r in all_results.items():
        if r.get('space_before_punct', 0) > 0:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                matches = re.finditer(r'(\S{{0,30}})\s+([,.;:!?])', text)
                count = 0
                for m in matches:
                    if count >= max_samples:
                        break
                    before = m.group(1).replace('\n', '\\n')
                    punct = m.group(2)
                    print(f"  Space before punct [{os.path.basename(filepath)}]: ...{before} {punct}")
                    count += 1
                if count > 0:
                    print()
            except:
                pass

        if r.get('double_spaces', 0) > 0:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                matches = re.finditer(r'(.{{0,30}})(  +)(.{{0,30}})', text)
                count = 0
                for m in matches:
                    if count >= max_samples:
                        break
                    print(f"  Double space [{os.path.basename(filepath)}]: ...{m.group(0).replace(chr(10), '\\n')}...")
                    count += 1
                if count > 0:
                    print()
            except:
                pass

    # Find garbage chars samples
    for filepath, r in all_results.items():
        if r.get('garbage_chars'):
            print(f"  Garbage in {os.path.basename(filepath)}: {r['garbage_chars']}")
    print()


if __name__ == '__main__':
    # Determine which directories to check
    data_dirs = []
    if len(sys.argv) > 1:
        data_dirs = sys.argv[1:]
    else:
        # Default directories
        data_dirs = ['books', 'data/books']

    # Filter to existing directories
    existing_dirs = [d for d in data_dirs if os.path.isdir(d)]
    if not existing_dirs:
        print(f"Error: No valid directories found among: {data_dirs}")
        sys.exit(1)

    print(f"Scanning directories: {', '.join(existing_dirs)}")
    print()

    all_results = {}

    for data_dir in existing_dirs:
        for root, dirs, files in os.walk(data_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)

                # Skip binary files (no extension) - check only .txt files
                if not fname.endswith('.txt'):
                    # Check if it might be a text file without extension
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            f.read(1024)
                    except:
                        continue

                try:
                    result = validate_file(fpath)
                    all_results[fpath] = result
                except Exception as e:
                    print(f"  Error processing {fpath}: {e}")

    if not all_results:
        print("No text files found to validate.")
        sys.exit(1)

    summarize_results(all_results, existing_dirs)
    show_sample_issues(all_results)
