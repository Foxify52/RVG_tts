import winsound
import torch, os, sys, numpy as np
from typing import Callable

from utils.checkpoints import init_tts_model
from utils.dsp import DSP
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer
from multiprocessing import cpu_count
from utils.vc_infer_pipeline import VC
from lib.rvc.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from fairseq import checkpoint_utils
from scipy.io import wavfile

now_dir = os.getcwd()
sys.path.append(now_dir)
hubert_model=None

class Synthesizer:

    def __init__(self,
                 tts_path: str,
                 device='cuda'):
        self.device = torch.device(device)
        tts_checkpoint = torch.load(tts_path, map_location=self.device)
        tts_config = tts_checkpoint['config']
        tts_model = init_tts_model(tts_config)
        tts_model.load_state_dict(tts_checkpoint['model'])
        self.tts_model = tts_model
        self.vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
        self.vocoder.to(device).eval()
        self.cleaner = Cleaner.from_config(tts_config)
        self.tokenizer = Tokenizer()
        self.dsp = DSP.from_config(tts_config)

    def __call__(self,
                 text: str,
                 voc_model: str,
                 alpha=1.0,
                 pitch_function: Callable[[torch.tensor], torch.tensor] = lambda x: x,
                 energy_function: Callable[[torch.tensor], torch.tensor] = lambda x: x,
                 ) -> np.array:
        x = self.cleaner(text)
        x = self.tokenizer(x)
        x = torch.tensor(x).unsqueeze(0)
        gen = self.tts_model.generate(x,
                                      alpha=alpha,
                                      pitch_function=pitch_function,
                                      energy_function=energy_function)
        m = gen['mel_post'].cpu()
        if voc_model == 'griffinlim':
            wav = self.dsp.griffinlim(m.squeeze().numpy(), n_iter=32)
        else:
            m = m.cuda()
            with torch.no_grad():
                wav = self.vocoder.inference(m).cpu().numpy()
        return wav

def pcm2float(sig, dtype='float32'):
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

class Config:
    def __init__(self,device,is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([f'{now_dir}\\models\\hubert.pt'],suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

def vc_single(sid, audio, input_audio, f0_up_key, f0_file, f0_method, file_index, index_rate):
    global tgt_sr,net_g,vc,hubert_model, version
    f0_up_key = int(f0_up_key)
    times = [0, 0, 0]
    if(hubert_model==None):
        load_hubert()
    if_f0 = cpt.get("f0", 1)
    audio_opt=vc.pipeline(
        model=hubert_model, 
        net_g=net_g, 
        sid=sid, 
        audio=audio, 
        input_audio_path=input_audio, 
        times=times, 
        f0_up_key=f0_up_key, 
        f0_method=f0_method,
        file_index=file_index,
        index_rate=index_rate,
        if_f0=if_f0,
        filter_radius=3,
        tgt_sr=tgt_sr,
        resample_sr=0,
        rms_mix_rate=0.25,
        version=version,
        protect=0.33,
        f0_file=f0_file
    )
    print(times)
    return audio_opt

def get_vc(model_path): # TODO update get_vc to work with v1 256 synth models without causing size mismatch error
    global n_spk,tgt_sr,net_g,vc,cpt,device,is_half, version
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

#Config
device = "cuda:0"
is_half = False
config=Config(device,is_half)

def rvg_tts(input_text):
    synth_forward = Synthesizer(tts_path=f'{now_dir}\\models\\forward.pt')
    synth_output = pcm2float(synth_forward(input_text, voc_model='melgan', alpha=1.2), dtype=np.float32)

    get_vc(f"{now_dir}\\models\\rvc_model.pth")
    wav_opt=vc_single(
        sid=0, 
        audio=synth_output,
        input_audio=None, 
        f0_up_key=0,
        f0_file=None, 
        f0_method="crepe",
        file_index=f"{now_dir}\\models\\rvc_index.index",
        index_rate=0.6,
    )
    wavfile.write("output.wav", tgt_sr, wav_opt)
    winsound.PlaySound("output.wav", winsound.SND_FILENAME)

rvg_tts('Sample text. Replace me!')