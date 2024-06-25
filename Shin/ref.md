__주제가 굉장히 비슷한 캐글__
https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data

__Audio Tokenizer 인데, Voice 에도 효과가 있을지는 의문.__
https://github.com/microsoft/unilm/tree/master/beats

__XGBoost 와 RandomForest 가 좋았다.__
https://arxiv.org/pdf/2308.12734v1

__굳이 딥러닝 안써도 됐다.__
https://openaccess.thecvf.com/content/CVPR2022W/WMF/papers/Borzi_Is_Synthetic_Voice_Detection_Research_Going_Into_the_Right_Direction_CVPRW_2022_paper.pdf


__wav2vec2이 제일 좋다, MLP랑, wav2vec2/xls-r-2b, Self Attention layer 모두 분류 층은 로지스틱 회귀가 성능이 좋았다. 왜냐면 선형 모델이라 정규화 역할을 한다. 추가 정규화는 더 악화됏음.__
https://arxiv.org/html/2309.05384v2

__SVM classifier using an RBF kernel 이걸 쓰면 전기 조금 들고 성능도 좋음.__
https://arxiv.org/html/2403.14290v1

__내가 다 검토했거든? SSL? 을 써라.__
https://www.mdpi.com/1999-4893/15/5/155

__custom cnn 썼음.__
https://www.sciencedirect.com/science/article/pii/S1877050923002910/pdf?md5=1c2d5fe4af33853a755190cd07c84008&pid=1-s2.0-S1877050923002910-main.pdf
