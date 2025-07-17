import re

def align_comments(lines):
    # Find the max length before comment
    max_len = 0
    for line in lines:
        if '#' in line:
            code, comment = line.split('#', 1)
            max_len = max(max_len, len(code.rstrip()))
        else:
            max_len = max(max_len, len(line.rstrip()))

    aligned_lines = []
    for line in lines:
        if '#' in line:
            code, comment = line.split('#', 1)
            aligned_line = code.rstrip().ljust(max_len + 2) + '# ' + comment.strip()
        else:
            aligned_line = line.rstrip()
        aligned_lines.append(aligned_line)
    return aligned_lines

# ========== Example ========== #
if __name__ == "__main__":
    # Read your code file
    file_path = "your_code.py"  # Replace this with the file you want to align
    with open(file_path, 'r') as f:
        lines = f.readlines()

    aligned = align_comments(lines)

    # Write back the aligned code
    with open(file_path, 'w') as f:
        f.write('\n'.join(aligned))
