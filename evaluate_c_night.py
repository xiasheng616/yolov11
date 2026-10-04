"""Evaluate saved YOLO ByteTrack labels against MOT17-04 GT on CPU.

Uses the existing 960x540 website preview, not the original image sequence.
The resulting scores are local preview-video results, not leaderboard scores.
"""
from pathlib import Path
import argparse
import configparser
import hashlib
import json
import math
import shutil
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
TOOLS = WORKSPACE / 'work' / 'tracking_eval'
sys.path.insert(0, str(TOOLS / 'python_deps'))
sys.path.insert(0, str(TOOLS / 'TrackEval-master'))


def convert_labels(label_dir, output_file, width=1920, height=1080, count=1050):
    files = list(label_dir.glob('C_night_*.txt'))
    assert len(files) == count, (label_dir, len(files), count)
    frames = {int(p.stem.rsplit('_', 1)[1]): p for p in files}
    assert set(frames) == set(range(1, count+1))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    input_hash = hashlib.sha256()
    with output_file.open('w', encoding='utf-8', newline='') as stream:
        for frame_no in range(1, count+1):
            contents = frames[frame_no].read_bytes()
            input_hash.update(frames[frame_no].name.encode()+b'\0'+contents+b'\0')
            ids = set()
            for line in contents.decode().splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                assert len(values) == 7 and all(math.isfinite(v) for v in values), (label_dir, frame_no, line)
                cls, cx, cy, bw, bh, conf, tid = values
                assert cls == 0 and tid == int(tid) and tid > 0 and tid not in ids
                assert bw > 0 and bh > 0 and 0 <= conf <= 1
                ids.add(tid)
                left, top = (cx-bw/2)*width+1, (cy-bh/2)*height+1
                stream.write(f'{frame_no},{int(tid)},{left:.6f},{top:.6f},{bw*width:.6f},{bh*height:.6f},{conf:.6f},-1,-1,-1\n')
                total += 1
    return {'label_files': len(files), 'boxes': total, 'input_sha256': input_hash.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=Path(r'E:\yolov11\yolov11\runs\detect'))
    parser.add_argument('--video', type=Path, default=Path(r'E:\yolov11\yolov11\data\videos\C_night.mp4'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    import cv2
    import numpy as np
    # TrackEval upstream still uses the old NumPy scalar aliases.
    # Restore their original builtin meanings without changing metric formulas.
    for alias, builtin in [('float', float), ('int', int), ('bool', bool)]:
        if alias not in np.__dict__:
            setattr(np, alias, builtin)

    seq = 'MOT17-04-DPM'
    names = ['baseline_v2', 'match07_v2', 'match09_v2', 'buffer60_v2']
    out = WORKSPACE / 'outputs' / 'mot17_04_evaluation'
    out.mkdir(parents=True, exist_ok=True)
    gt_sources = list(TOOLS.glob('**/MOT17-04-DPM/gt/gt.txt'))
    assert len(gt_sources) == 1, gt_sources
    source_seq = gt_sources[0].parents[1]
    ini = configparser.ConfigParser()
    ini.read(source_seq / 'seqinfo.ini')
    W, H = ini.getint('Sequence', 'imWidth'), ini.getint('Sequence', 'imHeight')
    N, FPS = ini.getint('Sequence', 'seqLength'), ini.getfloat('Sequence', 'frameRate')
    cap = cv2.VideoCapture(str(args.video))
    assert cap.isOpened(), args.video
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert (W, H, N) == (1920, 1080, 1050), (W, H, N)
    assert (vw, vh) == (960, 540) and abs(fps-FPS) < 0.01, (vw, vh, fps)
    sample_frames = {}
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if frame_count in (1, 200, 525, 800, 1050):
            sample_frames[frame_count] = frame
    cap.release()
    assert frame_count == N, (frame_count, N)

    gt_dir = out / 'gt' / 'MOT17-train' / seq
    (gt_dir / 'gt').mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_seq / 'seqinfo.ini', gt_dir / 'seqinfo.ini')
    shutil.copy2(gt_sources[0], gt_dir / 'gt' / 'gt.txt')
    seqmap = out / 'seqmap.txt'
    seqmap.write_text('name\n'+seq+'\n', encoding='utf-8')
    manifest = {'sequence': seq, 'video': str(args.video), 'video_sha256': hashlib.sha256(args.video.read_bytes()).hexdigest(),
                'video_width': vw, 'video_height': vh, 'decoded_frames': frame_count, 'fps': fps,
                'gt_width': W, 'gt_height': H, 'coordinate_convention': 'YOLO zero-origin normalized centers -> original-size boxes; +1 to left/top for MOT 1-based convention',
                'protocol': 'Local evaluation on compressed, downscaled website preview; official GT and TrackEval preprocessing; no leaderboard comparability',
                'runs': {}}
    for name in names:
        label_dir = args.runs / name / 'labels'
        dest = out / 'trackers' / 'MOT17-train' / name / 'data'
        manifest['runs'][name] = convert_labels(label_dir, dest / f'{seq}.txt', W, H, N)
        cfg_name = name.removesuffix('_v2')
        cfg = Path(r'C:\Users\19738\Documents\Codex\bytetrack_configs') / f'bytetrack_{cfg_name}.yaml'
        if cfg.exists():
            (out / 'configs').mkdir(exist_ok=True)
            shutil.copy2(cfg, out / 'configs' / cfg.name)
        print(name, 'converted boxes:', manifest['runs'][name]['boxes'], flush=True)

    # Visual alignment audit: show official pedestrian GT on sampled preview frames.
    gt = np.loadtxt(gt_sources[0], delimiter=',')
    audit = out / 'alignment'
    audit.mkdir(exist_ok=True)
    for frame_no, frame in sample_frames.items():
        rows = gt[(gt[:, 0] == frame_no) & (gt[:, 6] == 1) & (gt[:, 7] == 1)]
        for row in rows:
            x, y, w, h = row[2:6]
            p1 = (round((x-1)*vw/W), round((y-1)*vh/H))
            p2 = (round((x-1+w)*vw/W), round((y-1+h)*vh/H))
            cv2.rectangle(frame, p1, p2, (0, 220, 0), 1)
            cv2.putText(frame, str(int(row[1])), p1, cv2.FONT_HERSHEY_SIMPLEX, .35, (0,220,0), 1)
        cv2.putText(frame, f'MOT17-04 official GT / preview frame {frame_no}', (10,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        assert cv2.imwrite(str(audit / f'gt_frame_{frame_no:04d}.jpg'), frame)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.prepare_only:
        return

    import trackeval
    evaluator = trackeval.Evaluator({'USE_PARALLEL': False, 'PRINT_RESULTS': True,
        'PRINT_CONFIG': False, 'OUTPUT_SUMMARY': True, 'OUTPUT_DETAILED': True, 'PLOT_CURVES': False,
        'BREAK_ON_ERROR': True, 'LOG_ON_ERROR': str(out / 'errors.log')})
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(out / 'gt'), 'TRACKERS_FOLDER': str(out / 'trackers'),
        'OUTPUT_FOLDER': str(out / 'metrics'), 'TRACKERS_TO_EVAL': names,
        'BENCHMARK': 'MOT17', 'SPLIT_TO_EVAL': 'train', 'SEQMAP_FILE': str(seqmap),
        'CLASSES_TO_EVAL': ['pedestrian'], 'DO_PREPROC': True, 'PRINT_CONFIG': False})
    evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
    summary = {}
    for name in names:
        lines = (out / 'metrics' / name / 'pedestrian_summary.txt').read_text().splitlines()
        row = dict(zip(lines[0].split(), map(float, lines[1].split())))
        summary[name] = {k: row[k] for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW','CLR_FP','CLR_FN','CLR_Re','CLR_Pr']}
    (out / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
