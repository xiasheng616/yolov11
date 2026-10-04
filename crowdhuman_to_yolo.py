"""Convert CrowdHuman to a single-class Ultralytics YOLO practice dataset.

Run this file with the SERVER Python interpreter in PyCharm.
Ignore policy: omit mask/extra.ignore boxes from positive labels and preserve
them in metadata. Standard YOLO still sees their pixels as possible background;
this does NOT implement ignore-aware loss or official CrowdHuman evaluation.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT = Path('/root/autodl-tmp/yolov11/yolov11')
SOURCE = PROJECT / 'datasets/CrowdHuman'
OUTPUT = PROJECT / 'datasets/CrowdHuman_yolo'


def normalize_box(box, width, height):
    """Clip a pixel-space [x,y,w,h] box and return normalized xywh."""
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError(f'Non-finite box: {box}')
    if w <= 0 or h <= 0:
        return None
    x1, y1 = max(0.0, min(width, x)), max(0.0, min(height, y))
    x2, y2 = max(0.0, min(width, x + w)), max(0.0, min(height, y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def convert_split(split, expected_count):
    image_dir = OUTPUT / 'images' / split
    label_dir = OUTPUT / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    # Labels are regenerated, so remove only this dataset's label cache.
    label_dir.with_suffix('.cache').unlink(missing_ok=True)

    annotations = SOURCE / f'annotation_{split}.odgt'
    records = [json.loads(line) for line in annotations.read_text(
        encoding='utf-8-sig').splitlines() if line.strip()]
    if len(records) != expected_count:
        raise ValueError(f'{annotations}: expected {expected_count} records, '
                         f'found {len(records)}')
    ids = [record['ID'] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{split}: duplicate image IDs in annotations')

    image_list, ignored_records = [], []
    boxes_count = ignored_count = invalid_count = empty_count = 0
    for number, record in enumerate(records, 1):
        image_id = record['ID']
        if Path(image_id).name != image_id or '/' in image_id or '\\' in image_id:
            raise ValueError(f'Unexpected image ID: {image_id}')
        source_image = SOURCE / split / 'Images' / f'{image_id}.jpg'
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        # Decode the whole image rather than checking only its header.
        try:
            with Image.open(source_image) as im:
                im.load()
                width, height = im.size
                preview = im.convert('RGB') if number <= 3 else None
        except Exception as exc:
            raise RuntimeError(f'Unreadable image: {source_image}') from exc

        lines, ignored, seen_boxes = [], [], set()
        for box in record['gtboxes']:
            if box.get('tag') != 'person' or box.get('extra', {}).get('ignore', 0) == 1:
                ignored.append(box)
                ignored_count += 1
                continue
            coords = normalize_box(box['fbox'], width, height)
            if coords is None:
                invalid_count += 1
                continue
            line = '0 ' + ' '.join(f'{v:.8f}' for v in coords)
            if line in seen_boxes:
                continue
            seen_boxes.add(line)
            lines.append(line)
            if preview is not None:
                cx, cy, bw, bh = coords
                ImageDraw.Draw(preview).rectangle(
                    ((cx - bw/2) * width, (cy - bh/2) * height,
                     (cx + bw/2) * width, (cy + bh/2) * height),
                    outline='red', width=2)

        # One symlink per annotated image excludes notebook checkpoint files.
        target_image = image_dir / source_image.name
        if target_image.is_symlink():
            if target_image.readlink() != source_image:
                raise RuntimeError(f'Unexpected existing symlink: {target_image}')
        elif target_image.exists():
            raise RuntimeError(f'Output image already exists as a regular file: {target_image}')
        else:
            target_image.symlink_to(source_image)

        (label_dir / f'{image_id}.txt').write_text(
            '\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        image_list.append(str(target_image))
        boxes_count += len(lines)
        empty_count += not lines
        if ignored:
            ignored_records.append({'ID': image_id, 'gtboxes': ignored})
        if preview is not None:
            preview.save(OUTPUT / 'preview' / f'{split}_{number}.jpg')
        if number % 1000 == 0 or number == expected_count:
            print(f'{split}: {number}/{expected_count}', flush=True)

    (OUTPUT / f'{split}.txt').write_text(
        '\n'.join(image_list) + '\n', encoding='utf-8')
    (OUTPUT / 'metadata' / f'ignored_{split}.odgt').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ignored_records),
        encoding='utf-8')
    return {'images': len(image_list), 'positive_boxes': boxes_count,
            'omitted_ignore_boxes': ignored_count,
            'invalid_or_outside_boxes': invalid_count,
            'images_without_positive_boxes': empty_count}


def main():
    for split in ('train', 'val'):
        annotation = SOURCE / f'annotation_{split}.odgt'
        if not annotation.is_file():
            raise FileNotFoundError(f'Missing annotation: {annotation}')
    (OUTPUT / 'metadata').mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'preview').mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, count in [('train', 15000), ('val', 4370)]:
        stats[split] = convert_split(split, count)
        print(f'{split}: {stats[split]}', flush=True)

    config = OUTPUT / 'crowdhuman.yaml'
    config.write_text(
        f'path: {OUTPUT}\ntrain: train.txt\nval: val.txt\nnames:\n  0: person\n',
        encoding='utf-8')
    (OUTPUT / 'metadata' / 'conversion.json').write_text(
        json.dumps({'box_type': 'fbox', 'clip_boxes_to_image': True,
                    'ignore_policy': 'omit_positive_labels_only',
                    'ignore_aware_loss': False, 'stats': stats}, indent=2),
        encoding='utf-8')
    print(f'\nFinished. Dataset config: {config}')
    print(f'Check the red boxes in: {OUTPUT / "preview"}')
    print('WARNING: ignore boxes were omitted, not masked in the YOLO loss.')
    print('YOLO mAP is not an official CrowdHuman benchmark score.')


if __name__ == '__main__':
    main()
