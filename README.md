# RVG TTS
The retrieval based voice generation text to speech system is a python based text to speech that relies on two core parts. to be able to generate speech, It relies on tacotron to convert the text to speech and then uses rvc voice conversion to be able to make it sound like any character without the need to use an audio file.

## Requirements
This tts has been tested on [python 3.10](https://www.python.org/downloads/release/python-31011/) although might work on other versions.

You are required to have the latest 64 bit [Espeak NG](https://github.com/espeak-ng/espeak-ng/releases) release.

In order to build the fairseq dependency, you are required to have [Visual Studio](https://visualstudio.microsoft.com/downloads/) and install the "Desktop development with C++" development package.

## Usage
To use it, simply install the requirements with `pip install -r requirements.txt` and then download the [Hubert](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt) model, [Forward Tacotron](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ForwardTacotron/forward_step90k.pt) model and any [RVC](https://discord.com/invite/aihub) model.
You can then place them into the model folder with the corresponding names:
- `hubert_base.pt` -> `hubert.pt`
- `forward_steps90k.pt` -> `forward.pt`
- `(rvc .pth model name)` -> `rvc_model.pth`
- `(rvc .index model name)` -> `rvc_index.index` (optional)

Once you have all of these, you can run the `RVG.py` file with your desired arguments over CLI or you can include this code in your own project and import the `rvg_tts` function from `RVG.py`.

## Current feature set
 - RVC v1 and v2 model support
 - RVC Index support
 - Fast inference speed (~10 seconds on start and ~5 on consecutive runs with persistent mode on via importing)
 - Easy to use CLI

## Todo
 - [X] Support both RVC model versions
 - [X] Create a proper importable package
 - [X] Support calling from CLI
 - [X] Further code condensing
 - [ ] Multi-lang support

## Other languages
In order to use a different language, a new forward tacotron model must be trained. This is something I cannot do without a dataset. This is where I ask the community for help. If you can provide a dataset, please do.

## Credits
[Forward Tacotron](https://github.com/as-ideas/ForwardTacotron) is licensed under the [MIT License](https://github.com/as-ideas/ForwardTacotron/blob/master/LICENSE)

[RVC Webui](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) is licensed under the [MIT License](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)

## License
   Copyright 2023 Foxify52

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
