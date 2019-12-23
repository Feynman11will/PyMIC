# 数据预处理
- [ChannelWiseNormalize, Pad]
    - ChannelWiseNormalize
    - 分层求均值和方差，而后归一化
        - chn_norm = (image[chn] - chn_mean)/chn_std
    - Pad
        - 