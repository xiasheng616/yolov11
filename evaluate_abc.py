"""Compare four ByteTrack configurations on the A/B/C MOT17 preview videos."""
import argparse
import configparser
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

CHAT = Path(r'D:\Users\19738\Documents\Codex\2026-09-17\new-chat-2')
VARIANTS = ('baseline', 'match07', 'match09', 'buffer60')
SCENES = {'A': ('A_street', 'MOT17-09-DPM', 525),
          'B': ('B_crowd', 'MOT17-02-DPM', 600),
          'C': ('C_night', 'MOT17-04-DPM', 1050)}


def convert(folder, stem, dest, width, height, frames):
    paths = list(folder.glob(stem + '_*.txt'))
    if not paths:
        raise ValueError(f'Missing labels: {folder}')
    numbered = {}
    for path in paths:
        frame = int(path.stem.rsplit('_', 1)[1])
        if frame not in range(1, frames + 1) or frame in numbered:
            raise ValueError(f'Invalid frame number: {path}')
        numbered[frame] = path
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with dest.open('w', encoding='utf-8', newline='') as stream:
        for frame in range(1, frames + 1):
            # YOLO omits TXT files for frames with no reported tracks.
            if frame not in numbered:
                continue
            raw = numbered[frame].read_bytes()
            digest.update(numbered[frame].name.encode() + b'\0' + raw + b'\0')
            ids = set()
            for line in raw.decode('utf-8').splitlines():
                if not line.strip():
                    continue
                vals = list(map(float, line.split()))
                if len(vals) != 7 or not all(math.isfinite(v) for v in vals):
                    raise ValueError(f'Expected class cx cy w h confidence ID: {numbered[frame]}')
                cls, cx, cy, w, h, conf, tid = vals
                if cls != 0 or tid < 1 or tid != int(tid) or tid in ids or w <= 0 or h <= 0 or not 0 <= conf <= 1:
                    raise ValueError(f'Invalid label: {numbered[frame]}: {line}')
                ids.add(tid)
                x, y = (cx-w/2)*width+1, (cy-h/2)*height+1
                stream.write(f'{frame},{int(tid)},{x:.6f},{y:.6f},{w*width:.6f},{h*height:.6f},{conf:.6f},-1,-1,-1\n')
                count += 1
    return {'label_files': len(paths), 'frames_without_labels': frames-len(paths),
            'boxes': count, 'labels_sha256': digest.hexdigest()}


def prepare(args):
    manifest = {'protocol': 'Local scores on compressed 960x540 previews, not official original-image benchmark scores.',
                'coordinate_rule': 'normalized YOLO centers to original pixel boxes, +1 left/top for MOT convention', 'scenes': {}}
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.resources / 'MOT17Labels.zip') as archive:
        for letter, (stem, seq, expected) in SCENES.items():
            gt_dir = args.output / 'gt' / 'MOT17-train' / seq
            for relative in ('seqinfo.ini', 'gt/gt.txt'):
                suffix = f'/{seq}/{relative}'
                matches = [name for name in archive.namelist() if ('/'+name).endswith(suffix)]
                if len(matches) != 1:
                    raise ValueError(f'Missing or ambiguous official GT: {seq}/{relative}')
                target = gt_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(matches[0]))
            ini = configparser.ConfigParser()
            ini.read(gt_dir / 'seqinfo.ini')
            W, H, N = (ini.getint('Sequence', key) for key in ('imWidth','imHeight','seqLength'))
            fps = ini.getint('Sequence', 'frameRate')
            if (W,H,N,fps) != (1920,1080,expected,30):
                raise ValueError(f'Unexpected official metadata: {seq}')
            info = {'sequence': seq, 'stem': stem, 'frames': N, 'width': W, 'height': H, 'fps': fps, 'runs': {}}
            for variant in VARIANTS:
                run = f'{variant}_v2' if letter == 'C' else f'{letter}_{variant}'
                dest = args.output / 'trackers' / 'MOT17-train' / variant / 'data' / (seq+'.txt')
                info['runs'][variant] = convert(args.runs / run / 'labels', stem, dest, W,H,N)
                info['runs'][variant]['source_run'] = run
                print(f'{letter} {variant}: {info["runs"][variant]["boxes"]} boxes', flush=True)
            manifest['scenes'][letter] = info
    seqmap = args.output / 'seqmap.txt'
    seqmap.write_text('name\n'+'\n'.join(row[1] for row in SCENES.values())+'\n', encoding='utf-8')
    (args.output / 'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest, seqmap


def audit_videos(args, manifest, cv2, np):
    for letter, info in manifest['scenes'].items():
        video = args.videos / (info['stem']+'.mp4')
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w,h)!=(960,540) or abs(cap.get(cv2.CAP_PROP_FPS)-info['fps'])>.01:
            raise ValueError(f'Unexpected preview resolution or FPS: {video}')
        gt = np.loadtxt(args.output/'gt'/'MOT17-train'/info['sequence']/'gt'/'gt.txt',delimiter=',')
        audit = args.output/'alignment'/letter
        audit.mkdir(parents=True,exist_ok=True)
        selected={1,(info['frames']+1)//2,info['frames']}
        count=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:
                    break
                count+=1
                if count not in selected:
                    continue
                for row in gt[(gt[:,0]==count)&(gt[:,6]==1)&(gt[:,7]==1)]:
                    x,y,bw,bh=row[2:6]
                    p1=(round((x-1)*w/info['width']),round((y-1)*h/info['height']))
                    p2=(round((x-1+bw)*w/info['width']),round((y-1+bh)*h/info['height']))
                    cv2.rectangle(frame,p1,p2,(0,220,0),1)
                    cv2.putText(frame,str(int(row[1])),p1,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,220,0),1)
                cv2.putText(frame,f'{info["sequence"]} GT / frame {count}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                if not cv2.imwrite(str(audit/f'gt_frame_{count:04d}.jpg'),frame):
                    raise ValueError('Could not save alignment image')
        finally:
            cap.release()
        if count!=info['frames']:
            raise ValueError(f'Frame count mismatch: {letter}: {count} vs {info["frames"]}')
        info['decoded_frames']=count
        info['video_sha256']=hashlib.sha256(video.read_bytes()).hexdigest()


def summarize(raw, np):
    h,c,i=raw['HOTA'],raw['CLEAR'],raw['Identity']
    return {'HOTA':round(float(np.mean(h['HOTA']))*100,3),
            'IDF1':round(float(i['IDF1'])*100,3),'MOTA':round(float(c['MOTA'])*100,3),
            'IDSW':int(c['IDSW']),'FP':int(c['CLR_FP']),'FN':int(c['CLR_FN'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path(r'E:\yolov11\yolov11\runs\detect'))
    parser.add_argument('--videos',type=Path,default=Path(r'E:\yolov11\yolov11\data\videos'))
    parser.add_argument('--resources',type=Path,default=CHAT/'work'/'tracking_eval')
    parser.add_argument('--output',type=Path,default=CHAT/'outputs'/'mot17_abc_evaluation')
    parser.add_argument('--convert-only',action='store_true')
    args=parser.parse_args()
    manifest,seqmap=prepare(args)
    if args.convert_only:
        print('All 12 runs validated and converted. Output:',args.output,flush=True)
        return
    import cv2
    import numpy as np
    # Compatibility for upstream TrackEval's old NumPy scalar names; formulas unchanged.
    for alias,builtin in [('float',float),('int',int),('bool',bool)]:
        if alias not in np.__dict__:
            setattr(np,alias,builtin)
    audit_videos(args,manifest,cv2,np)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sys.path.insert(0,str(args.resources/'TrackEval-master'))
    import trackeval
    evaluator=trackeval.Evaluator({'USE_PARALLEL':False,'PRINT_RESULTS':False,'PRINT_CONFIG':False,
        'OUTPUT_SUMMARY':True,'OUTPUT_DETAILED':True,'PLOT_CURVES':False,'BREAK_ON_ERROR':True,
        'LOG_ON_ERROR':str(args.output/'errors.log')})
    dataset=trackeval.datasets.MotChallenge2DBox({'GT_FOLDER':str(args.output/'gt'),
        'TRACKERS_FOLDER':str(args.output/'trackers'),'OUTPUT_FOLDER':str(args.output/'metrics'),
        'TRACKERS_TO_EVAL':list(VARIANTS),'BENCHMARK':'MOT17','SPLIT_TO_EVAL':'train',
        'SEQMAP_FILE':str(seqmap),'CLASSES_TO_EVAL':['pedestrian'],'DO_PREPROC':True,'PRINT_CONFIG':False})
    result,_=evaluator.evaluate([dataset],[trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),trackeval.metrics.Identity({'PRINT_CONFIG':False})])
    results=result[dataset.get_name()]
    summary={}
    lines=['# MOT17 preview-video comparison','',
           'Local compressed/downscaled-preview results; no direct leaderboard comparison.',
           'COMBINED uses TrackEval aggregation, not a simple mean of per-video scores.','']
    for letter in (*SCENES,'COMBINED'):
        seq='COMBINED_SEQ' if letter=='COMBINED' else SCENES[letter][1]
        summary[letter]={v:summarize(results[v][seq]['pedestrian'],np) for v in VARIANTS}
        print(f'\n{letter}: HOTA / IDF1 / MOTA / IDSW / FP / FN',flush=True)
        lines += [f'## {letter}','','| Config | HOTA | IDF1 | MOTA | IDSW | FP | FN |','|---|---:|---:|---:|---:|---:|---:|']
        for v,row in summary[letter].items():
            print(f'{v:10s} {row["HOTA"]:8.3f} {row["IDF1"]:8.3f} {row["MOTA"]:8.3f} {row["IDSW"]:6d} {row["FP"]:7d} {row["FN"]:7d}',flush=True)
            lines.append(f'| {v} | {row["HOTA"]:.3f} | {row["IDF1"]:.3f} | {row["MOTA"]:.3f} | {row["IDSW"]} | {row["FP"]} | {row["FN"]} |')
        lines.append('')
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (args.output/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\nSaved comparison.json and comparison.md to:',args.output,flush=True)


if __name__=='__main__':
    main()
