import os
import sys
from datetime import datetime
import tensorflow as tf
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("TensorFlowTTS/")
from TensorFlowTTS.tensorflow_tts.inference import AutoConfig
from TensorFlowTTS.tensorflow_tts.inference import TFAutoModel
from TensorFlowTTS.tensorflow_tts.inference import AutoProcessor

import scipy.io.wavfile as wavf

class VoiceSynthesis:
    # 모델 초기화 
    def __init__(self):

        # tacotron 설정, 학습된 모델 가져오기
        module_path = os.path.dirname(os.path.abspath(__file__))        
        tacotron2_config = AutoConfig.from_pretrained('./TensorFlowTTS/examples/tacotron2/conf/tacotron2.kss.v1.yaml')
        self.tacotron2 = TFAutoModel.from_pretrained(
            config=tacotron2_config,
            pretrained_path="./tacotron2-100k.h5",
            name="tacotron2"
        )

        # mel gan 설정, 학습된 모델 가져오기
        mb_melgan_config = AutoConfig.from_pretrained('./TensorFlowTTS/examples/multiband_melgan/conf/multiband_melgan.v1.yaml')
        self.mb_melgan = TFAutoModel.from_pretrained(
            config=mb_melgan_config,
            pretrained_path="./melgan-1000k.h5",
            name="mb_melgan"
        )

        #processor - 글자 별 상응하는 숫자의 mapper 설정 가져오기
        self.processor = AutoProcessor.from_pretrained(pretrained_path="kss_mapper.json")

    # 입력 text -> 음성 변환 함수
    def do_synthesis(self, input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
        # 문자(초,중,종성) -> 숫자 sequence 변환 
        input_ids = self.processor.text_to_sequence(input_text)

        # text2mel part
        if text2mel_name == "TACOTRON":
            _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
                tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                tf.convert_to_tensor([len(input_ids)], tf.int32),
                tf.convert_to_tensor([0], dtype=tf.int32)
            )
        else:
            raise ValueError("Only TACOTRON, FASTSPEECH2 are supported on text2mel_name")

        # vocoder part
        if vocoder_name == "MB-MELGAN":
            audio = vocoder_model.inference(mel_outputs)[0, :, 0]
        else:
            raise ValueError("Only MB_MELGAN are supported on vocoder_name")

        if text2mel_name == "TACOTRON":
            return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
        else:
            return mel_outputs.numpy(), audio.numpy()

    # attention graph 출력
    def visualize_attention(self, alignment_history):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_title(f'Alignment steps')
        im = ax.imshow(
            alignment_history,
            aspect='auto',
            origin='lower',
            interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        plt.xlabel(xlabel)
        plt.ylabel('Encoder timestep')
        plt.tight_layout()
        plt.show()
        plt.close()

    # mel spectrogram 시각화 
    def visualize_mel_spectrogram(self, mels):
        mels = tf.reshape(mels, [-1, 80]).numpy()
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(311)
        ax1.set_title(f'Predicted Mel-after-Spectrogram')
        im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
        fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
        plt.show()
        plt.close()
    '''어텐션그래프와 멜스펙트로그램 시각화는 jpg로 첨부 '''
    
    def text_to_voice(self,input_text):
        # 현재시간을 파일 제목으로 사용
        cur_time = datetime.now()
        timestamp_str = cur_time.strftime("%Y%m%d_%H%M%S_%f")
        # audio 절대 경로에 생성
        mels, alignment_history, audios = self.do_synthesis(input_text, self.tacotron2, self.mb_melgan, "TACOTRON", "MB-MELGAN")
        # mels, audios = self.do_synthesis(input_text, self.fastspeech2, self.mb_melgan, "FASTSPEECH2", "MB-MELGAN")
        sample_rate = 22050
        # audio가 저장될 위치 - ./output/
        output_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)),'output',timestamp_str +'.wav')
        wavf.write(output_audio, sample_rate, audios)
        return output_audio
    
    # flask app 용 
    def tts_flask(self,input_text):
        # 현재시간을 파일 제목으로 사용
        cur_time = datetime.now()
        timestamp_str = cur_time.strftime("%H%M")
        # audio 절대 경로에 생성
        mels, alignment_history, audios = self.do_synthesis(input_text, self.tacotron2, self.mb_melgan, "TACOTRON", "MB-MELGAN")
        sample_rate = 22050
        # audio가 저장될 위치 - ./output/
        output_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)),timestamp_str +'.wav')
        wavf.write(output_audio, sample_rate, audios)
        return timestamp_str



if __name__ == "__main__":

    tts = VoiceSynthesis()
    start = time.time()  # 시작 시간 저장
    input_text = "안녕하세요 코드스테이츠 3기입니다"
    print(tts.text_to_voice(input_text))
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
