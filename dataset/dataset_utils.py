import os
from pathlib import Path
import shutil
import torchvision

def maybe_cache_file(path: os.PathLike):
    '''Motivation: if every job reads from a shared disk it`ll get very slow, consider an image can
    be 2MB, then with batch size 32, 16 workers in dataloader you`re already requesting 1GB!! -
    imagine this for all users and all jobs simultaneously.'''
    # checking if we are on cluster, not on a local machine
    if 'LOCAL_SCRATCH' in os.environ:
        cache_dir = os.environ.get('LOCAL_SCRATCH')
        # a bit ugly but we need not just fname to be appended to `cache_dir` but parent folders,
        # otherwise the same fnames in multiple folders will create a bug (the same input for multiple paths)
        cache_path = os.path.join(cache_dir, Path(path).relative_to('/'))
        if not os.path.exists(cache_path):
            os.makedirs(Path(cache_path).parent, exist_ok=True)
            shutil.copyfile(path, cache_path)
        return cache_path
    else:
        return path

def get_video_and_audio(path, get_meta=False, start_sec=0, end_sec=None):
    orig_path = path
    path = maybe_cache_file(path)
    # (Tv, 3, H, W) [0, 255, uint8]; (Ca, Ta)
    rgb, audio, meta = torchvision.io.read_video(str(path), start_sec, end_sec, 'sec', output_format='TCHW')
    assert meta['video_fps'], f'No video fps for {orig_path}'
    # (Ta) <- (Ca, Ta)
    audio = audio.mean(dim=0)
    # FIXME: this is legacy format of `meta` as it used to be loaded by VideoReader.
    meta = {'video': {'fps': [meta['video_fps']]}, 'audio': {'framerate': [meta['audio_fps']]}, }
    return rgb, audio, meta

