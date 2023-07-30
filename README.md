# RVG TTS
The retrieval based voice generation text to speech system is a python based text to speech that relies on two core parts. to be able to generate speech, It relies on tacotron to convert the text to speech and then uses rvc voice conversion to be able to make it sound like any character without the need to use an audio file.

## Usage
To use it, simply install the requirements with `pip install -r requirements.txt` and then download the [Hubert](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt) model, [Forward Tacotron](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ForwardTacotron/forward_step90k.pt) model and any [RVC v2](https://discord.com/invite/aihub) model.
You can then place them into the model folder with the corresponding names:
- `hubert_base.pt` -> `hubert.pt`
- `forward_steps90k.pt` -> `forward.pt`
- `(rvc v2 .pth model name)` -> `rvc_model.pth`
- `(rvc v2 .index model name)` -> `rvc_index.index` (optional)

Once you have all of these, you can run the `RVG.py` file and replace the sample text on the provided function call at the bottom or you can include this code in your own project and import the `rvg_tts` function from `RVG.py`.

## Todo
 - [ ] Add in support for the RVC v1 models
 - [X] Fix assertion error with faiss index when using rvc index files
 - [ ] Create a proper importable package
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