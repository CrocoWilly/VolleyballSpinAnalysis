from pathlib import Path

paths = [
    'datasets/ball_detection/conti_volleyball_label_1015/labels',
    'datasets/ball_detection/volleyball_label_tvl19/1/labels',
    'datasets/ball_detection/volleyball_label_tvl19/2/labels',
    'datasets/ball_detection/volleyball_label_tvl19/3/labels',
    'datasets/ball_detection/volleyball_label_tvl19/4/labels',
    'datasets/ball_detection/volleyball_label_tvl19/5/labels',
    'datasets/ball_detection/volleyball_label_tvl19/6/labels',
    'datasets/ball_detection/volleyball_label_1014/labels',
]

def fix_dataset(path):
    checked = 0
    fixed = 0
    path = Path(path)
    for file in path.rglob('*.txt'):
        if file.is_dir():
            continue
        if file.stem == 'classes':
            continue
        new_lines = []
        is_need_fix = False
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_toks = line.split(' ')
                if len(line_toks) != 5:
                    print(f'Ignore {file}')
                    continue
                x, y, w, h = map(float, line_toks[1:])
                if x < 0:
                    print(f'x < 0 in {file}')
                    x = 0
                    is_need_fix = True
                if y < 0:
                    print(f'y < 0 in {file}')
                    y = 0
                    is_need_fix = True
                if w < 0:
                    print(f'w < 0 in {file}')
                    w = 0
                    is_need_fix = True
                if h < 0:
                    print(f'h < 0 in {file}')
                    h = 0
                    is_need_fix = True
                if x + w > 0.999999:
                    print(f'x + w ({x + w}) > 1 in {file}')
                    w = 0.999998 - x
                    is_need_fix = True
                if y + h > 0.999999:
                    print(f'y + h ({y + h}) > 1 in {file}')
                    h = 0.999998 - y
                    is_need_fix = True
                line_toks[1:] = map(lambda v: f'{v:.6f}', [x, y, w, h])
                new_line = ' '.join(line_toks)
                new_lines.append(new_line)
        if is_need_fix:
            fixed += 1
            with open(file, 'w') as f:
                f.write('\n'.join(new_lines))
        checked += 1
    print(f'Checked {checked} files in {path}')
    print(f'\tFixed {fixed} files in {path}')
    return checked, fixed
                
total_checked = 0
total_fixed = 0
for path in paths:
    checked, fixed = fix_dataset(path)
    total_checked += checked
    total_fixed += fixed
print(f'Total checked {total_checked} files')
print(f'Total fixed {total_fixed} files')