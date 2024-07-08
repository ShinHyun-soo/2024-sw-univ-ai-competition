요 전략대로 한번 해봅시다.

현재 hubert-deepfake  0.3577    
wav2vec deepfake      0.37xx    
단순히 평균냄.         0.32xx    

각자 하나씩 붙잡고 최대한 단일 모델 성능을 높인다음에 stacking 앙상블이 뭔진 모르겟지만 해봅시다.

 https://dacon.io/competitions/official/236105/codeshare/8435    
HuggingFace에서 다음의 모델을 fine-tuning하여 stacking 앙상블 하였습니다    
facebook/wav2vec2-base    
MIT/ast-finetuned-audioset-10-10-0.4593    
facebook/wav2vec2-conformer-rope-large-960h-ft    
facebook/hubert-xlarge-ll60k    
microsoft/unispeech-sat-large    
microsoft/wavlm-large     
facebook/wav2vec2-conformer-rel-pos-large    
microsoft/unispeech-large-1500h-cv    
facebook/data2vec-audio-large    
asapp/sew-mid-100k    



## 학습 과정   
> [허깅페이스 링크(오디오 사전학습 모델을 deepfake 탐지 데이터로 파인튜닝한 모델들](https://huggingface.co/models?other=audio-classification&sort=downloads&search=deep)   
> 위에서 맘에 드는 것을 고른 후,(업로드 날짜가 최신인데, 다운로드 수가 많은 모델을 먼저 해보는 것 추천)      
> [파인 튜닝 및 앙상블 코드 - 코드 해석 난이도 상](https://dacon.io/competitions/official/236105/codeshare/8431)    
> [파인튜닝 코드 - 코드 해석 난이도 중](https://dacon.io/competitions/official/236105/codeshare/8435)     
> [파인튜닝 코드 - 코드 해석 난이도 하](https://dacon.io/competitions/official/236105/codeshare/8426)     
> 위 코드 중 맘에 드는 것을 골라    
> 데이콘 데이터 셋으로 파인튜닝 진행,      
> 카톡방이나 여기에 결과 공유 ( 상세할 수록 좋음)     
--   
## 공유 예시   
> Noise 추가 증강기법 이용하여 motheecreator/Deepfake-audio-detection 에 파인튜닝함.
> kfold 10, validation 주기 0.1, batch_size = 16, 등등
> Dacon accuracy = 0.85 


