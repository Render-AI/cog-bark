from typing import Optional
from scipy.io.wavfile import write as write_wav
from cog import BasePredictor, Input, Path, BaseModel
from bark import SAMPLE_RATE, generate_audio, preload_models, save_as_prompt
from bark.generation import ALLOWED_PROMPTS
import nltk  # we'll use this to split into sentences
from voicefixer import VoiceFixer
from voicefixer import Vocoder

class ModelOutput(BaseModel):
    prompt_npz: Optional[Path]
    audio_out: Path


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # for the pushed version on Replicate, the CACHE_DIR from bark/generation.py is changed to a local folder to
        # include the weights file in the image for faster inference
        nltk.download('punkt')
        preload_models()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Hello, I am, uh — how should I say this... [laughs] I am a voice clone made by an A.I.",
        ),
        history_prompt: str = Input(
            description="history choice for audio cloning, choose from the list",
            default=None,
            choices=sorted(list(ALLOWED_PROMPTS)),
        ),
        custom_history_prompt: Path = Input(
            description="Provide your own .npz file with history choice for audio cloning, this will override the "
            "previous history_prompt setting",
            default=None,
        ),
        text_temp: float = Input(
            description="semantic token generation temperature (1.0 more diverse, 0.0 more conservative)",
            default=0.7,
        ),
        waveform_temp: float = Input(
            description="coarse audio token generation temperature (1.0 more diverse, 0.0 more conservative)",
            default=0.7,
        ),
        waveform_fine_temp: float = Input(
            description="fine audio token generation temperature (1.0 more diverse, 0.0 more conservative)",
            default=0.7,
        ),
        output_full: bool = Input(
            description="return full generation as a .npz file to be used as a history prompt", default=False
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        if custom_history_prompt is not None:
            history_prompt = str(custom_history_prompt)

        '''
        # begin original output function
        audio_array = generate_audio(
            prompt,
            history_prompt=history_prompt,
            text_temp=text_temp,
            waveform_temp=waveform_temp,
            output_full=output_full,
        )
        output = "/tmp/audio.wav"
        if not output_full:
            write_wav(output, SAMPLE_RATE, audio_array)
            return ModelOutput(audio_out=Path(output))
        out_npz = "/tmp/prompt.npz"
        save_as_prompt(out_npz, audio_array[0])
        write_wav(output, SAMPLE_RATE, audio_array[-1])
        # end of original output function
        '''

        # Imports adapted from https://github.com/gemelo-ai/vocos/blob/main/notebooks%2FBark%2BVocos.ipynb
        import torch
        from vocos import Vocos
        from typing import Optional, Union, Dict
        import numpy as np
        from bark.generation import generate_coarse, generate_fine

        # Source: https://github.com/gemelo-ai/vocos/blob/main/notebooks%2FBark%2BVocos.ipynb
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocos = Vocos.from_pretrained(
            "charactr/vocos-encodec-24khz").to(device)

        def semantic_to_audio_tokens(
            semantic_tokens: np.ndarray,
            history_prompt: history_prompt,
            temp: float = 0.7,
            fine_temp: float = 0.5,
            silent: bool = False,
            output_full: bool = True,
        ):
            coarse_tokens = generate_coarse(
                semantic_tokens, history_prompt=history_prompt, temp=temp, silent=silent, use_kv_caching=True
            )
            fine_tokens = generate_fine(
                coarse_tokens, history_prompt=history_prompt, temp=fine_temp)

            if output_full:
                full_generation = {
                    "semantic_prompt": semantic_tokens,
                    "coarse_prompt": coarse_tokens,
                    "fine_prompt": fine_tokens,
                }
                # print("Full generation: \n")
                # print(full_generation)
                return full_generation
            return fine_tokens

        from bark import text_to_semantic

        # Longform generation code taken from https://github.com/gitmylo/bark-data-gen/blob/main/notebooks/long_form_generation.ipynb
        from IPython.display import Audio
        import torchaudio

        prompt = prompt.replace("\n", " ").strip()

        sentences = nltk.sent_tokenize(prompt)
        SPEAKER = history_prompt
        # silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
        count = 1
        audioFiles = []
        finalOutput = None
        previousPrompt = None

        for sentence in sentences:
            # generate with Vocos
            print("Sentence: \n" + sentence)
            if previousPrompt is not None:
                history_prompt = previousPrompt
            text_prompt = " " + sentence + " "
            # TODO: feed previous sentence into text_to_semantic by generating history_prompt dynamically
            semantic_tokens = text_to_semantic(
                text_prompt, history_prompt=history_prompt, temp=text_temp, silent=False,)
            audio_tokens = semantic_to_audio_tokens(
                semantic_tokens, history_prompt=history_prompt, temp=waveform_temp, fine_temp=waveform_fine_temp, silent=False, output_full=True,
            )

            # print("Audio tokens: \n")
            # print(audio_tokens)

            previousPrompt = audio_tokens

            from bark.generation import codec_decode

            from IPython.display import Audio

            encodec_output = codec_decode(audio_tokens['fine_prompt'])

            import torchaudio
            # Upsample to 44100 Hz for better reproduction on audio hardware
            encodec_output = torchaudio.functional.resample(
                torch.from_numpy(encodec_output), orig_freq=24000, new_freq=44100)
            Audio(encodec_output, rate=44100)

            audio_tokens_torch = torch.from_numpy(
                audio_tokens['fine_prompt']).to(device)
            features = vocos.codes_to_features(audio_tokens_torch)
            vocos_output = vocos.decode(
                features, bandwidth_id=torch.tensor([2], device=device))  # 6 kbps
            # Upsample to 44100 Hz for better reproduction on audio hardware
            vocos_output = torchaudio.functional.resample(
                vocos_output, orig_freq=24000, new_freq=44100).cpu()
            Audio(vocos_output.numpy(), rate=44100)
            filename = f"temp-output-{count}.mp3"
            torchaudio.save(
                filename, encodec_output[None, :], 44100, compression=128)            
            audioFiles.append(filename)

            print(f"Exporting {filename}...\n")
            from pydub import AudioSegment
            if finalOutput is None:
                finalOutput = AudioSegment.from_mp3(filename)
            if finalOutput is not None:                
                currentOutput = AudioSegment.from_mp3(filename)
                finalOutput = finalOutput.append(currentOutput, crossfade=50) 
            if(count == len(sentences)):
                print("Exporting combined output...\n")
                print(audioFiles) #printing the array        
                finalOutput.export("output.mp3", format="mp3")
            count = count + 1                
            # end code for generation with Vocos
        
        # voicefixer = VoiceFixer()
        # Mode 0
        VoiceFixer.restore(input="./output.mp3", # input file path (wav or mp3)
                   output="./final-output.mp3", # output file path (wav or mp3)
                   cuda=True, # whether to use gpu acceleration
                   mode = 0) # You can try out mode 0, 1, 2 to find out the best result -- 0 is the default

        return ModelOutput(audio_out=Path('final-output.mp3'))
