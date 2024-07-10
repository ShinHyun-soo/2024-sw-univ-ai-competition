## [전략 2024-07-10 14:55 Ver.]

### 배경 지식
> Submission 을 합치니까 결과가 좋았음. 아마 어딘가로 회귀하는 것 같음.               
> 합쳐보니, 기존의 모델은 real 확률이 크게 나옴.      
> 근데? fake 기준으로 학습을 시키면? 그나마 fake 가 크게 나옴 (거의 real 이 큼)      
> 따라서 기존 모델은 real 확률이 큰게 대부분이니까 이걸 평균을 내서 깎아야함 (fake 확률이 크게 fake 기준으로 학습을 해서)    
> 근데? 합칠 때 모델 자체가 다른 것(hubert-deepfake, hubert-large 이건 같은 모델, hubert-large, wav2vec-deepfake 이건 다른 모델, 앞에 뭐라고 써있는지 확인하는게 중요., 저 아래 목록은 그냥 대충 찾아놓은거임. 저대로 학습하라는게 아님. 저 모델들 기반에 deepfake 로 fine-tuning 된 거를 찾아보라는 의미, 근데 찾아보니까 wavlm, wav2vec 만 잇엇음.) 끼리 합치는게 좋았음.       

### 결론
> 현재, hubert 기반의 real 기준으로 학습되어, real 확률이 높은게 많음. (test 자체가 real 이 많은 거일 수도 있음. 근데 real 을 낮추고 fake 를 올리는게 성능이 향상됨. 잘 모르겟음.)        
> __따라서, wavlm 이나 wav2vec deepfake 모델을 라벨 바꾸기 잡기술을 이용해서, fake 기준 확률이 조금이라도 높게 학습을 해서 나온 submission 을 합치는게 좋다고 보임.__      

### 할 것
> github 코드 참고해서 wav2vec 이나 wavlm 을 deep-fake 로 finetuning 된 모델을(깃헙 코드 보면 잇음.) fake 기준으로 학습을 한 ckpt 가지고 submission 을 생성해서 카톡방에 업로드.    

### 팁
> fold 2 에 val_interval = 1, epoch 는 0 ( 1번만 학습) 하는게 뭔가 좋은것 같음. 이유는 모름.       




                      











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


