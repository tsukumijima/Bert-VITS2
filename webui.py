# flake8: noqa: E402
import os
import logging
import re_matching
from tools.sentence import split_by_language

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import gradio as gr
import webbrowser
import numpy as np
from config import config
import librosa

net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    style_text,
    style_weight,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    style_text,
    style_weight,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer_multilang(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
    reference_audio,
    emotion,
    style_text,
    style_weight,
):
    if style_text == "":
        style_text = None
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    audio_list = []
    if not cut_by_sent:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio = infer(
                p,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                style_text=style_text,
                style_weight=style_weight,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
    else:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
            for idx, s in enumerate(sent_list):
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sent_list) - 1) and skip_end
                audio = infer(
                    s,
                    reference_audio=reference_audio,
                    emotion=emotion,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                    style_text=style_text,
                    style_weight=style_weight,
                    skip_start=skip_start,
                    skip_end=skip_end,
                )
                audio_list_sent.append(audio)
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (44100, audio_concat))


def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    prompt_mode,
    style_text=None,
    style_weight=0,
):
    if style_text == "":
        style_text = None
    if prompt_mode == "Audio prompt":
        if reference_audio == None:
            return ("Invalid audio prompt", None)
        else:
            reference_audio = load_audio(reference_audio)[1]
    else:
        reference_audio = None
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        result = []
        for slice in re_matching.text_matching(text):
            _speaker = slice.pop()
            temp_contant = []
            temp_lang = []
            for lang, content in slice:
                if "|" in content:
                    temp = []
                    temp_ = []
                    for i in content.split("|"):
                        if i != "":
                            temp.append([i])
                            temp_.append([lang])
                        else:
                            temp.append([])
                            temp_.append([])
                    temp_contant += temp
                    temp_lang += temp_
                else:
                    if len(temp_contant) == 0:
                        temp_contant.append([])
                        temp_lang.append([])
                    temp_contant[-1].append(content)
                    temp_lang[-1].append(lang)
            for i, j in zip(temp_lang, temp_contant):
                result.append([*zip(i, j), _speaker])
        for i, one in enumerate(result):
            skip_start = i != 0
            skip_end = i != len(result) - 1
            _speaker = one.pop()
            idx = 0
            while idx < len(one):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    lang, content = one[idx]
                    temp_text = [content]
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(one):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(one) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        _speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        style_text,
                        style_weight,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    elif language.lower() == "auto":
        for idx, slice in enumerate(text.split("|")):
            if slice == "":
                continue
            skip_start = idx != 0
            skip_end = idx != len(text.split("|")) - 1
            sentences_list = split_by_language(
                slice, target_languages=["zh", "ja", "en"]
            )
            idx = 0
            while idx < len(sentences_list):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    content, lang = sentences_list[idx]
                    temp_text = [content]
                    lang = lang.upper()
                    if lang == "JA":
                        lang = "JP"
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(sentences_list):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sentences_list) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        style_text,
                        style_weight,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    else:
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
        )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)


def load_audio(path):
    audio, sr = librosa.load(path, 48000)
    # audio = librosa.resample(audio, 44100, 48000)
    return sr, audio


def gr_util(item):
    if item == "Text prompt":
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }
    else:
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }


if __name__ == "__main__":
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["JP"]
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    text = gr.TextArea(
                        label="読み上げテキスト",
                        placeholder="""Bert-VITS2 は、読み上げテキストが示す感情に合わせて、自動的に抑揚や感情表現を調整します。
例えば「ありがとうございます！」なら前向きに明るい声で、「とても残念です…。」なら残念そうな声で読み上げできます。
句読点の有無や、！？… などの文末記号の使い方次第で、抑揚や感情表現が大きく変わります。
意図した表現にならないときは、読み上げテキストを工夫してみてください。

Bert-VITS2 は行の最初の方の文から読み取れる感情表現を後まで引き継ぐ傾向があるため、まとまった文ごとに改行で区切り、[音声を行ごとに分割して生成] ボタンを押すとより自然な音声を生成できます。
ただし、「とても嬉しいです。しかし残念です。」のように真逆の感情を含む文では、同じ行に含めた方がより自然な繋がりになることもあります。""",
                    )
                    # slicer = gr.Button("クイックスライス", variant="primary")
                    speaker = gr.Dropdown(
                        choices=speakers, value=speakers[0], label="Speaker"
                    )
                    language = gr.Dropdown(
                        choices=languages, value=languages[0], label="Language"
                    )
                with gr.Group():
                    sdp_ratio = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.1, label="抑揚の強さ (SDP Ratio): 0.2 ~ 0.6 の範囲がおすすめです。0 にすると棒読みになります。"
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise"
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1, maximum=2, value=0.8, step=0.1, label="Noise_W"
                    )
                    length_scale = gr.Slider(
                        minimum=0.1, maximum=2, value=1.0, step=0.1, label="Length"
                    )
            with gr.Column():
                with gr.Accordion("テキストプロンプト / 音声プロンプト (オプション)", open=False):
                    gr.Markdown(
                        value="指定されたテキストプロンプト / 音声プロンプトのスタイルを持つ音声を生成できます。<br>"
                        "**現状日本語ではほとんど効果がありません。読み上げテキスト自体の感情表現を工夫した方が、より効果的です。**"
                    )
                    prompt_mode = gr.Radio(
                        ["Text prompt", "Audio prompt"],
                        label="Prompt Mode",
                        value="Text prompt",
                    )
                    text_prompt = gr.Textbox(
                        label="Text prompt",
                        placeholder="スタイルの Prompt (例: Happy)",
                        value="",
                        visible=True,
                    )
                    audio_prompt = gr.Audio(
                        label="Audio prompt", type="filepath", visible=False
                    )
                with gr.Accordion("テキストセマンティクスの融合 (オプション)", open=False):
                    gr.Markdown(
                        value="補助テキストのセマンティクスを使用して合成時の抑揚や感情表現を調整できます。実際に読み上げられる文章は読み上げテキストと同じです。<br>"
                        "**注意: 指示/宣言的なテキスト (例: 幸せ) は使用せず、強い感情を込めたテキスト (例: とても幸せです！！) を使用してください。**\n\n"
                        "効果はあまり明確ではありません。空白のままにしておくと、この機能は使用されません。<br>"
                        "読み上げテキストの発音に誤りがある場合、正しい発音の同音異字に置き換えてみてください。同時に、元の読み上げテキストをここに記入し、Weight を最大にすることで正しい発音を得ることができます。これにより、元のテキストの Bert の意味情報も保持されます。"
                    )
                    style_text = gr.Textbox(label="補助テキスト")
                    style_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.7,
                        step=0.1,
                        label="Weight",
                        info="Bert の読み上げテキストと補助テキストの混合比率 / 0 は読み上げテキストのみ、1 は補助テキストのみを意味します。",
                    )
                with gr.Row():
                    with gr.Column():
                        interval_between_para = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=0.4,
                            step=0.1,
                            label="行と行の間での一時停止 (秒): [文と文の間での一時停止] よりも長い秒数に設定すると効果的です。",
                        )
                        interval_between_sent = gr.Slider(
                            minimum=0,
                            maximum=5,
                            value=0.2,
                            step=0.1,
                            label="文と文の間での一時停止 (秒): [文ごとに分割して生成] にチェックを入れたときのみ有効です。",
                        )
                        opt_cut_by_sent = gr.Checkbox(
                            label="一文ごとに分割して生成: 行ごとの分割に加えて、さらに読み上げテキストを一文ごとに分割して生成します。"
                        )
                        slicer = gr.Button("音声を行ごとに分割して生成 (おすすめ)", variant="primary")
                        btn = gr.Button("音声を分割せずに生成")
                text_output = gr.Textbox(label="ステータス:", visible=False)
                audio_output = gr.Audio(label="音声出力")
                # explain_image = gr.Image(
                #     label="参数解释信息",
                #     show_label=True,
                #     show_share_button=False,
                #     show_download_button=False,
                #     value=os.path.abspath("./img/参数说明.png"),
                # )
        btn.click(
            tts_fn,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                audio_prompt,
                text_prompt,
                prompt_mode,
                style_text,
                style_weight,
            ],
            outputs=[text_output, audio_output],
        )

        slicer.click(
            tts_split,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                opt_cut_by_sent,
                interval_between_para,
                interval_between_sent,
                audio_prompt,
                text_prompt,
                style_text,
                style_weight,
            ],
            outputs=[text_output, audio_output],
        )

        prompt_mode.change(
            lambda x: gr_util(x),
            inputs=[prompt_mode],
            outputs=[text_prompt, audio_prompt],
        )

        audio_prompt.upload(
            lambda x: load_audio(x),
            inputs=[audio_prompt],
            outputs=[audio_prompt],
        )

    print("The inference page is available.")
    # webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")
    print(f'Please visit http://127.0.0.1:{config.webui_config.port} to use the Web UI.')
    app.launch(share=config.webui_config.share, server_name='0.0.0.0', server_port=config.webui_config.port)
