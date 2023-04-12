import os
import json
import torch
import shutil
import random
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip, AudioFileClip


class Preprocessor:
    def __init__(self, data_dir: str, output_dir:str,audio_sr: int, video_fps: int):
        self.audio_sr = audio_sr
        self.video_fps = video_fps
        self.video_dir = os.path.join(data_dir, "video")
        self.audio_dir = os.path.join(data_dir, "audio")
        self.output_data_dir = os.path.join(output_dir, "processed")
        # Audio feature extractor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.audio_extractor_bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        self.audio_extractor = (
            self.audio_extractor_bundle.get_model().eval().float().to(self.device)
        )

    def preprocess(self):
        """
        - Each datapoint should have:
            - Audio frame feature is extracted using
            - Target frame (the frame to be predicted)
        - Folder structure:
            - Video / Audio dir:
                - Speaker 1:
                    - Video Folder:
                        - Video Files
                    - Audio Folder:
                        - Audio Files
                - Speaker 2:
                    etc.
        """
        # Create output directory
        if not os.path.exists(self.output_data_dir):
            os.makedirs(self.output_data_dir)
        # Iterate through each speaker
        for speaker in tqdm(os.listdir(self.video_dir), desc="Preprocessing"):
            # Iterate through each video file
            for video_folder in tqdm(
                os.listdir(os.path.join(self.video_dir, speaker)),
                "Video Folder",
                leave=False,
            ):
                for video_file in tqdm(
                    os.listdir(os.path.join(self.video_dir, speaker, video_folder)),
                    "Video File",
                    leave=False,
                ):
                    # Create save directory
                    video_file_name = video_file.replace(".mp4", "")

                    os.makedirs(
                        f"{self.output_data_dir}/video/{speaker}/{video_folder}/{video_file_name}",
                        exist_ok=True,
                    )
                    os.makedirs(
                        f"{self.output_data_dir}/audio/{speaker}/{video_folder}/{video_file_name}",
                        exist_ok=True,
                    )
                    os.makedirs(
                        f"{self.output_data_dir}/datapoints/{speaker}/{video_folder}/{video_file_name}",
                        exist_ok=True,
                    )
                    # Load video file
                    video = VideoFileClip(
                        os.path.join(self.video_dir, speaker, video_folder, video_file)
                    )
                    self.video_fps = int(video.fps)
                    # Load audio file
                    audio, sr = librosa.load(
                        os.path.join(
                            self.audio_dir,
                            speaker,
                            video_folder,
                            video_file.replace(".mp4", ".m4a"),
                        ),
                        sr=self.audio_sr,
                        mono=True,
                    )

                    # Get video frames
                    video_frames = video.iter_frames()
                    # Crop video frames to match audio frames
                    video_frames = [frame for frame in video_frames]
                    # Get audio frames
                    audio_frames = self.get_audio_frames(audio, len(video_frames))
                    audio_frames = audio_frames[: len(video_frames)]
                    video_frames = video_frames[: len(audio_frames)]
                    # Save each frame in output_folder
                    for i, video_frame in enumerate(video_frames):
                        # Save video frame
                        video_frame_path = os.path.join(
                            f"{self.output_data_dir}/video/{speaker}/{video_folder}/{video_file_name}",
                            f"{video_file.replace('.mp4', '')}_{i}.npy",
                        )
                        np.save(video_frame_path, np.transpose(video_frame, (2, 0, 1)))
                        # Save audio frame
                        audio_frame_path = os.path.join(
                            f"{self.output_data_dir}/audio/{speaker}/{video_folder}/{video_file_name}",
                            f"{video_file.replace('.mp4', '')}_{i}_audio.npy",
                        )
                        np.save(audio_frame_path, audio_frames[i])
                        if i > 1:
                            # Get motion frames
                            motion_frames = [
                                os.path.join(
                                    f"{self.output_data_dir}/video/{speaker}/{video_folder}/{video_file_name}",
                                    f"{video_file.replace('.mp4', '')}_{i-2}.npy",
                                ),
                                os.path.join(
                                    f"{self.output_data_dir}/video/{speaker}/{video_folder}/{video_file_name}",
                                    f"{video_file.replace('.mp4', '')}_{i-1}.npy",
                                ),
                            ]
                        else:
                            motion_frames = []
                        # Save data
                        data = {
                            "speaker": speaker,
                            "motion_frame": motion_frames,
                            "video_frame": video_frame_path,
                            "audio_frame": audio_frame_path,
                        }
                        with open(
                            os.path.join(
                                f"{self.output_data_dir}/datapoints/{speaker}/{video_folder}/{video_file_name}",
                                f"{video_file.replace('.mp4', '')}_{i}.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(data, f)

    def get_audio_frames(self, audio: np.array, n_video_frames:int):
        """
        Get audio frames from audio file
        """
        # Get audio frames
        # Convert audio features to correct format
        audio = torchaudio.functional.resample(
            torch.tensor(audio),
            self.audio_sr,
            self.audio_extractor_bundle.sample_rate,
        ).to(self.device)
        with torch.no_grad():
            audio_features = self.audio_extractor(audio.unsqueeze(0).float())
        audio_features = audio_features[0].cpu().numpy().squeeze(0)            
        # Calculate number of chunks
        # Split array into chunks
        audio_chunks = np.array_split(audio_features, n_video_frames)
        return audio_chunks


class AudioVideoFrameDataset(Dataset):
    def __init__(self, data_dir: str, frame_transforms=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.datapoints = [
            f"{data_dir}/datapoints/{spk}/{d}/{vid}/{datapoint}"
            for spk in os.listdir(f"{data_dir}/datapoints/")
            for d in os.listdir(f"{data_dir}/datapoints/{spk}")
            for vid in os.listdir(f"{data_dir}/datapoints/{spk}/{d}")
            for datapoint in os.listdir(f"{data_dir}/datapoints/{spk}/{d}/{vid}")
        ]
        self.frame_transforms = frame_transforms

    def __len__(self):
        # Calculate all subfolders in the video frames directory
        return len(self.datapoints)

    def __getitem__(self, idx):
        # Calcualte speaker and frame number
        with open(self.datapoints[idx], "r") as f:
            data = json.load(f)

        audio_frame = torch.from_numpy(np.load(data["audio_frame"]))
        video_frame = torch.from_numpy(np.load(data["video_frame"]))
        # Randomly select a frame from the video frames directory
        dir_path = "/".join(data["video_frame"].split("/")[:-1])
        id_frame = f'{dir_path}/{random.choice(os.listdir(dir_path))}'
        id_frame = torch.from_numpy(np.load(id_frame))
        # Get motion frames
        if data["motion_frame"]:
            motion_frames = [
                torch.from_numpy(np.load(frame)) for frame in data["motion_frame"]
            ]
        else:
            motion_frames = [id_frame, id_frame]
        # Apply transforms
        if self.frame_transforms:
            video_frame = self.frame_transforms(video_frame/255)
            id_frame = self.frame_transforms(id_frame/255)
            motion_frames = [self.frame_transforms(frame/255) for frame in motion_frames]

        return {
            "target_frame": video_frame,  # noise is added to this frame
            "audio_frame": audio_frame,  # this is the condition
            "motion_frames": motion_frames,  # this is also a condition that is concatenated to the noise frame
            "id_frame": id_frame,  # this is also a condition that is concatenated to the noise frame
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/j/Desktop/Programming/DeepLearning/multilingual/avatar/data/vox",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/vox",
        help="Path to data directory",
    )
    parser.add_argument(
        "--audio_sr",
        type=int,
        default=44100,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=25,
        help="Video frames per second",
    )
    args = parser.parse_args()

    # if os.path.exists(f"{args.output_dir}/processed"):
    #     action = input("Data already processed. Delete and re-run? (y/n): ").lower()
    #     if action == "y":
    shutil.rmtree(f"{args.output_dir}/processed")
    #     else:
    #         exit(0)

    preprocessor = Preprocessor(args.data_dir,args.output_dir, args.audio_sr, args.video_fps)
    preprocessor.preprocess()


    # dataset = AudioVideoFrameDataset('./data/vox/processed')

    # for i in range(10):
    #     outs = dataset[i]
    #     print(outs['target_frame'].shape)
    #     print(outs['audio_frame'][0].shape)
    #     print(outs['motion_frames'][0].shape)
    #     print(outs['id_frame'].shape)
        