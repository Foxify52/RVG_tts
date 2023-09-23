import torch, os, sys, argparse, winsound, numpy as np
from typing import Callable

from lib.fwt.dsp import DSP
from lib.fwt.forward_tacotron import ForwardTacotron
from lib.fwt.text_utils import Cleaner, Tokenizer
from multiprocessing import cpu_count
from lib.rvc.vc_infer_pipeline import VC
from lib.rvc.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
from fairseq import checkpoint_utils
from scipy.io import wavfile

class Synthesizer:

    def __init__(self,
                 tts_path: str,
                 device='cuda'):
        self.device = torch.device(device)
        tts_checkpoint = torch.load(tts_path, map_location=self.device)
        tts_config = tts_checkpoint['config']
        tts_model = ForwardTacotron.from_config(tts_config)
        tts_model.load_state_dict(tts_checkpoint['model'])
        self.tts_model = tts_model
        self.vocoder = torch.hub.load('seungwonpark/melgan', 'melgan', verbose=False)
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
        if voc_model == 'melgan':
            m = m.cuda()
            with torch.no_grad():
                wav = self.vocoder.inference(m).cpu().numpy()
        else:
            print("Specified vocoder isn't supported")
            exit()
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
        if torch.cuda.is_available() and self.device != "cpu":
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
            self.is_half = False

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

def vc_single(sid, audio, f0_up_key, f0_file, file_index, index_rate):
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
        times=times, 
        f0_up_key=f0_up_key, 
        file_index=file_index,
        index_rate=index_rate,
        if_f0=if_f0,
        tgt_sr=tgt_sr,
        resample_sr=0,
        rms_mix_rate=0.25,
        version=version,
        protect=0.5,
        f0_file=f0_file
    )
    print(times)
    return audio_opt

def get_vc(model_path):
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

def rvg_tts(
        input_text="hello world!",
        voice_transform=0, 
        tts_model=f"{os.getcwd()}\\models\\forward.pt",
        rvc_model=f"{os.getcwd()}\\models\\rvc_model.pth",
        rvc_index=f"{os.getcwd()}\\models\\rvc_index.index",
        device="cuda:0",
        is_half=True,
        silent_mode=False,
        persist=True
    ):
    global now_dir, config, hubert_model
    now_dir = os.getcwd()
    sys.path.append(now_dir)
    config=Config(device,is_half)
    hubert_model = None if persist else vars().get('hubert_model', None)

    synth_forward = Synthesizer(tts_model)
    synth_output = pcm2float(synth_forward(input_text, voc_model='melgan', alpha=1.3), dtype=np.float32)

    get_vc(rvc_model)
    wav_opt=vc_single(
        sid=0, 
        audio=synth_output,
        f0_up_key=voice_transform,
        f0_file=None,
        file_index=rvc_index,
        index_rate=0.6,
    )
    wavfile.write("output.wav", tgt_sr, wav_opt)
    if silent_mode == False:
        winsound.PlaySound("output.wav", winsound.SND_FILENAME)

parser = argparse.ArgumentParser(description = "A retrieval based voice generation text to speech system")
parser.add_argument("--input_text", default="hello world!", type=str, help="The input text to be converted to speech")
parser.add_argument("--voice_transform", default=0, type=int, help="The voice transposition to be applied (Ranges from -12 to 12)")
parser.add_argument("--tts_model", default=f"{os.getcwd()}\\models\\forward.pt", type=str, help="The path to the text-to-speech model")
parser.add_argument("--rvc_model", default=f"{os.getcwd()}\\models\\rvc_model.pth", type=str, help="The path to the RVC model")
parser.add_argument("--rvc_index", default=f"{os.getcwd()}\\models\\rvc_index.index", type=str, help="The path to the RVC index")
parser.add_argument("--device", default="cuda:0", type=str, help="The device to run the models on")
parser.add_argument("--is_half", action="store_false", help="Whether to use half precision for the models")
parser.add_argument("--silent_mode", action="store_false", help="Whether to suppress the output sound")
args = parser.parse_args()
rvg_tts(**vars(args))