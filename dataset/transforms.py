import logging
import math
import random
from typing import Tuple
import torch
import torchvision
import torchaudio
import numpy as np
import einops


def sec2frames(sec, fps):
    return int(sec * fps)

def frames2sec(frames, fps):
    return frames / fps


class RGBSpatialCrop(torch.nn.Module):
    def __init__(self, input_size, is_random):
        super().__init__()
        assert input_size is not None, f'smaller_input_size is `{input_size}`'
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.is_random = is_random

    @staticmethod
    def get_random_crop_sides(vid, output_size):
        '''Slice parameters for random crop'''
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    @staticmethod
    def get_center_crop_sides(vid, output_size):
        '''Slice parameters for center crop'''
        h, w = vid.shape[-2:]
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def forward(self, item):
        # (Tv, C, H, W)
        vid = item['video']
        if self.is_random:
            i, j, h, w = self.get_random_crop_sides(vid, self.input_size)
        else:
            i, j, h, w = self.get_center_crop_sides(vid, self.input_size)
        item['video'] = vid[..., i:(i + h), j:(j + w)]
        return item

class Resize(torchvision.transforms.Resize):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, item):
        item['video'] = super().forward(item['video'])
        return item


class RGBSpatialCropSometimesUpscale(torch.nn.Module):
    '''This (randomly) crops the input video and with prob `sometimes_p` this crop is smaller but upscaled
    to `target_input_size`'''

    def __init__(self, sometimes_p, target_input_size, is_random, smaller_input_size=None):
        super().__init__()
        self.sometimes_p = sometimes_p
        self.do_sometimes_upscale = sometimes_p is not None and sometimes_p > 0

        self.crop_only = RGBSpatialCrop(target_input_size, is_random)

        if self.do_sometimes_upscale:
            self.crop_further_and_upscale = torchvision.transforms.Compose([
                RGBSpatialCrop(smaller_input_size, is_random),
                Resize(target_input_size, antialias=None),
            ])

    def forward(self, item):
        assert len(item['video'].shape) == 4, \
            f"{item['video'].shape}: if it is applied after GenerateMultipleClips," \
            "augs should be applied to each clip separately, not to the whole video array. " \
            "Otherwise, ignore this warning (comment it)."
        if self.do_sometimes_upscale and self.sometimes_p > torch.rand(1):
            return self.crop_further_and_upscale(item)
        else:
            return self.crop_only(item)


class GenerateMultipleSegments(torch.nn.Module):
    '''
    Given an item with video and audio, generates a batch of `n_segments` segments
    of length `segment_size_vframes` (if None, the max number of segments will be made).
    If `is_start_random` is True, the starting position of the 1st segment will be random but respecting
    n_segments.
    `audio_jitter_sec` is the amount of audio offset in seconds.
    '''

    def __init__(self, segment_size_vframes: int, n_segments: int = None, is_start_random: bool = False,
                 audio_jitter_sec: float = 0., step_size_seg: float = 1):
        super().__init__()
        self.segment_size_vframes = segment_size_vframes
        self.n_segments = n_segments
        self.is_start_random = is_start_random
        self.audio_jitter_sec = audio_jitter_sec
        self.step_size_seg = step_size_seg
        logging.info(f'Segment step size: {self.step_size_seg}')

    def forward(self, item):
        v_len_frames, C, H, W = item['video'].shape
        a_len_frames = item['audio'].shape[0]

        v_fps = 25
        a_fps = 16000

        ## Determining the number of segments
        # segment size
        segment_size_vframes = self.segment_size_vframes
        segment_size_aframes = sec2frames(frames2sec(self.segment_size_vframes, v_fps), a_fps)

        # step size (stride)
        stride_vframes = int(self.step_size_seg * segment_size_vframes)
        stride_aframes = int(self.step_size_seg * segment_size_aframes)

        # Calculate the minimum required number of frames
        n_segments = self.n_segments if self.n_segments is not None else 1
        required_v_len = (n_segments - 1) * stride_vframes + segment_size_vframes
        required_a_len = (n_segments - 1) * stride_aframes + segment_size_aframes

        required_len_sec = frames2sec(required_v_len, v_fps)

        if v_len_frames < required_v_len:
            pad_v = required_v_len - v_len_frames
            pad_a = sec2frames(frames2sec(pad_v, v_fps), a_fps)
            
            pad_v_tensor = torch.zeros(pad_v, C, H, W, dtype=item['video'].dtype)
            item['video'] = torch.cat([item['video'], pad_v_tensor], dim=0)
            v_len_frames = required_v_len
            
            pad_a_tensor = torch.zeros(pad_a, dtype=item['audio'].dtype)
            item['audio'] = torch.cat([item['audio'], pad_a_tensor], dim=0)
            a_len_frames += pad_a

        if a_len_frames < required_a_len:
            pad_a = required_a_len - a_len_frames
            pad_a_tensor = torch.zeros(pad_a, dtype=item['audio'].dtype)
            item['audio'] = torch.cat([item['audio'], pad_a_tensor], dim=0)
            a_len_frames += pad_a

        # calculating the number of segments. (W - F + 2P) / S + 1
        n_segments_max_v = math.floor((v_len_frames - segment_size_vframes) / stride_vframes) + 1
        n_segments_max_a = math.floor((a_len_frames - segment_size_aframes) / stride_aframes) + 1

        # making sure audio and video can accommodate the same number of segments
        n_segments_max = min(n_segments_max_v, n_segments_max_a)
        n_segments = n_segments_max if self.n_segments is None else self.n_segments

        assert n_segments <= n_segments_max, \
            f'cant make {n_segments} segs of len {self.segment_size_vframes} in a vid ' \
            f'of len {v_len_frames} for {item["path"]}'

        # (n_segments, 2) each
        v_ranges, a_ranges = self.get_sequential_seg_ranges(v_len_frames, a_len_frames, v_fps, a_fps,
                                                            n_segments, segment_size_aframes)

        # segmenting original streams (n_segments, segment_size_frames, C, H, W)
        item['video'] = torch.stack([item['video'][s:e] for s, e in v_ranges], dim=0)
        item['audio'] = torch.stack([item['audio'][s:e] for s, e in a_ranges], dim=0)
        return item

    def get_sequential_seg_ranges(self, v_len_frames, a_len_frames, v_fps, a_fps, n_seg, seg_size_aframes):
        # if is_start_random is True, the starting position of the 1st segment will
        # be random but respecting n_segments like so: "-CCCCCCCC---" (maybe with fixed overlap),
        # else the segments are taken from the middle of the video respecting n_segments: "--CCCCCCCC--"

        seg_size_vframes = self.segment_size_vframes  # for brevity

        # calculating the step size in frames
        step_size_vframes = int(self.step_size_seg * seg_size_vframes)
        step_size_aframes = int(self.step_size_seg * seg_size_aframes)

        # calculating the length of the sequence of segments (and in frames)
        seg_seq_len = n_seg * self.step_size_seg + (1 - self.step_size_seg)
        vframes_seg_seq_len = int(seg_seq_len * seg_size_vframes)
        aframes_seg_seq_len = int(seg_seq_len * seg_size_aframes)

        # doing temporal crop
        max_v_start_i = v_len_frames - vframes_seg_seq_len
        if self.is_start_random:
            v_start_i = random.randint(0, max_v_start_i)
        else:
            v_start_i = max_v_start_i // 2
        a_start_i = sec2frames(frames2sec(v_start_i, v_fps), a_fps)  # vid frames -> seconds -> aud frames

        max_a_start_i = max(a_len_frames - aframes_seg_seq_len, 0)
        a_start_i = min(a_start_i, max_a_start_i)

        # make segments starts
        v_start_seg_i = torch.tensor([v_start_i + i * step_size_vframes for i in range(n_seg)]).int()
        a_start_seg_i = torch.tensor([a_start_i + i * step_size_aframes for i in range(n_seg)]).int()

        # # apply jitter to audio (original)
        # if self.audio_jitter_sec > 0:
        #     jitter_aframes = sec2frames(self.audio_jitter_sec, a_fps)
        #     # making sure after applying jitter, the audio is still within the audio boundaries
        #     jitter_aframes = min(jitter_aframes, a_start_i, a_len_frames-a_start_i-aframes_seg_seq_len)
        #     a_start_seg_i += random.randint(-jitter_aframes, jitter_aframes)  # applying jitter to segments

        # Edit
        if self.audio_jitter_sec > 0:
            jitter_aframes = sec2frames(self.audio_jitter_sec, a_fps)
            new_a_start_seg_i = []

            for start in a_start_seg_i:
                max_jitter = min(jitter_aframes, start.item(), a_len_frames - start.item() - seg_size_aframes)
                jitter = random.randint(-max_jitter, max_jitter)
                new_a_start_seg_i.append(start + jitter)

            a_start_seg_i = torch.tensor(new_a_start_seg_i, dtype=torch.int32)

        # make segment ends
        v_ends_seg_i = v_start_seg_i + seg_size_vframes
        a_ends_seg_i = a_start_seg_i + seg_size_aframes  # using the adjusted a_start_seg_i (with jitter)

        # make ranges
        v_ranges = torch.stack([v_start_seg_i, v_ends_seg_i], dim=1)
        a_ranges = torch.stack([a_start_seg_i, a_ends_seg_i], dim=1)
        assert (a_ranges >= 0).all() and (a_ranges <= a_len_frames).all(), f'{a_ranges} out of {a_len_frames}'
        assert (v_ranges <= v_len_frames).all(), f'{v_ranges} out of {v_len_frames}'
        return v_ranges, a_ranges

class RGBToFloatToZeroOne(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, item):
        item['video'] = item['video'].to(torch.float32).div(255.)
        return item


class RGBToHalfToZeroOne(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, item):
        item['video'] = item['video'].half().div(255.)
        return item


class RGBNormalize(torchvision.transforms.Normalize):
    '''The same as the torchvision`s but with different interface for the dict.
    This should work for any shape (..., C, H, W)'''

    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)
        logging.info(f'RGBNormalize: mean={mean}, std={std}')

    def forward(self, item):
        item['video'] = super().forward(item['video'])
        item['meta']['video']['norm_stats'] = {'mean': torch.as_tensor(self.mean),
                                               'std': torch.as_tensor(self.std)}
        return item


class AudioMelSpectrogram(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(**kwargs)

    def forward(self, item):
        item['audio'] = self.spec(item['audio'])  # safe for batched input
        return item


class AudioLog(torch.nn.Module):

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, item):
        item['audio'] = torch.log(item['audio'] + self.eps)
        return item

class PadOrTruncate(torch.nn.Module):

    def __init__(self, max_spec_t: int, pad_mode: str = 'constant', pad_value: float = 0.0):
        super().__init__()
        self.max_spec_t = max_spec_t
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def forward(self, item):
        item['audio'] = self.pad_or_truncate(item['audio'])
        return item

    def pad_or_truncate(self, audio):
        difference = self.max_spec_t - audio.shape[-1]  # safe for batched input
        # pad or truncate, depending on difference
        if difference > 0:
            # pad the last dim (time) -> (..., n_mels, 0+time+difference)  # safe for batched input
            pad_dims = (0, difference)
            audio = torch.nn.functional.pad(audio, pad_dims, self.pad_mode, self.pad_value)
        elif difference < 0:
            logging.warning(f'Truncating spec ({audio.shape}) to max_spec_t ({self.max_spec_t}).')
            audio = audio[..., :self.max_spec_t]  # safe for batched input
        return audio


class AudioNormalizeAST(torch.nn.Module):
    '''Normalization is done with two specified mean and std (half)'''
    def __init__(self, mean: float, std: float) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, item):
        item['audio'] = (item['audio'] - self.mean) / (2 * self.std)
        item['meta']['audio']['norm_stats'] = {'mean': self.mean, 'std': self.std}
        return item


class PermuteStreams(torch.nn.Module):

    def __init__(self, einops_order_audio: str, einops_order_rgb: str) -> None:
        ''' For example:
                einops_order_audio: "S F T -> S T F"
                einops_order_rgb: "S T C H W -> S C T H W"'''
        super().__init__()
        self.einops_order_audio = einops_order_audio
        self.einops_order_rgb = einops_order_rgb

    def forward(self, item):
        if self.einops_order_audio is not None:
            item['audio'] = einops.rearrange(item['audio'], self.einops_order_audio).contiguous()
        if self.einops_order_rgb is not None:
            item['video'] = einops.rearrange(item['video'], self.einops_order_rgb).contiguous()
        return item
