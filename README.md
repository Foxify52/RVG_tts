# RVG TTS
The retrieval based voice generation text to speech system is a python based text to speech that relies on two core parts. to be able to generate speech, It relies on tacotron to convert the text to speech and then uses rvc voice conversion to be able to make it sound like any character without the need to use an audio file.

## Requirements
This tts has been tested on [python 3.10](https://www.python.org/downloads/release/python-31011/) although might work on other versions.

You are required to have the latest 64 bit [Espeak NG](https://github.com/espeak-ng/espeak-ng/releases) release.

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
 - Fast inference speed (~10 seconds)

## Todo
 - [X] Support both RVC model versions
 - [ ] Create a proper importable package
 - [X] Support calling from CLI
 - [ ] Further code condensing 

## Credits
[Forward Tacotron](https://github.com/as-ideas/ForwardTacotron) is licensed under the [MIT License](https://github.com/as-ideas/ForwardTacotron/blob/master/LICENSE)

[RVC Webui](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) is licensed under the [MIT License](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)

## License
Retrieval based voice generation text to speech
Copyright (C) 2023  Foxify52

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.