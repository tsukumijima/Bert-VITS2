from onnx_modules.V220_OnnxInference import OnnxInferenceSession
import numpy as np
Session = OnnxInferenceSession(
        {
        "enc" : "onnx/BertVits2.2PT/BertVits2.2PT_enc_p.onnx",
        "emb_g" : "onnx/BertVits2.2PT/BertVits2.2PT_emb.onnx",
        "dp" : "onnx/BertVits2.2PT/BertVits2.2PT_dp.onnx",
        "sdp" : "onnx/BertVits2.2PT/BertVits2.2PT_sdp.onnx",
        "flow" : "onnx/BertVits2.2PT/BertVits2.2PT_flow.onnx",
        "dec" : "onnx/BertVits2.2PT/BertVits2.2PT_dec.onnx"
        },
        Providers = ["CPUExecutionProvider"]
    )

#����������ԭ����һ���ģ�ֻ��Ҫ��ԭ��Ԥ����������֮�����.numpy()����
x = np.expand_dims(
    np.array(
        [
            0,
            97,
            0,
            8,
            0,
            78,
            0,
            8,
            0,
            76,
            0,
            37,
            0,
            40,
            0,
            97,
            0,
            8,
            0,
            23,
            0,
            8,
            0,
            74,
            0,
            26,
            0,
            104,
            0,
        ]
    ),
    0
)
tone = np.zeros_like(x)
language = np.zeros_like(x)
sid = np.array([0])
bert = np.random.randn(x.shape[1], 1024)
ja_bert = np.random.randn(x.shape[1], 1024)
en_bert = np.random.randn(x.shape[1], 1024)
emo = np.random.randn(512, 1)

audio = Session(
    x,
    tone,
    language,
    bert,
    ja_bert,
    en_bert,
    emo,
    sid
)

print(audio)
