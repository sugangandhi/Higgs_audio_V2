from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message
import torch
import torchaudio
import os

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

def main():
    os.makedirs("results/higgs_v2", exist_ok=True)

    system_prompt = (
        "Generate high-quality, natural speech in a quiet studio.\n"
        "<|scene_desc_start|>\n"
        "Audio is recorded in a neutral environment with clear voice.\n"
        "<|scene_desc_end|>"
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(
            role="user",
            content="Hi, this is Sugan testing Higgs Audio V2 locally for our project. "
                    "Please speak clearly and naturally, with a friendly tone."
        ),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    serve_engine = HiggsAudioServeEngine(
        MODEL_PATH,
        AUDIO_TOKENIZER_PATH,
        device=device,
    )

    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )

    out_path = "results/higgs_v2/basic_test.wav"
    torchaudio.save(out_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
