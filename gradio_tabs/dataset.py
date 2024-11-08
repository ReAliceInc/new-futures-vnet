import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    time_suffix: bool,
    input_dir: str,
):
    if model_name == "":
        return "Error: を入力してください。"
    logger.info("Start slicing...")
    cmd = [
        "slice.py",
        "--model_name",
        model_name,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
    ]
    if time_suffix:
        cmd.append("--time_suffix")
    if input_dir != "":
        cmd += ["--input_dir", input_dir]
    # onnxの警告が出るので無視する
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "音声のスライスが完了しました。"


def do_transcribe(
    model_name,
    whisper_model,
    compute_type,
    language,
    initial_prompt,
    use_hf_whisper,
    batch_size,
    num_beams,
    hf_repo_id,
):
    if model_name == "":
        return "Error: モデル名を入力してください。"

    cmd = [
        "transcribe.py",
        "--model_name",
        model_name,
        "--model",
        whisper_model,
        "--compute_type",
        compute_type,
        "--language",
        language,
        "--initial_prompt",
        f'"{initial_prompt}"',
        "--num_beams",
        str(num_beams),
    ]
    if use_hf_whisper:
        cmd.append("--use_hf_whisper")
        cmd.extend(["--batch_size", str(batch_size)])
        if hf_repo_id != "openai/whisper":
            cmd.extend(["--hf_repo_id", hf_repo_id])
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}. エラーメッセージが空の場合、何も問題がない可能性があるので、書き起こしファイルをチェックして問題なければ無視してください。"
    return "音声の文字起こしが完了しました。"


def create_dataset_app() -> gr.Blocks:
    with gr.Blocks(theme=GRADIO_THEME) as app:
        # Markdownの定義

        gr.Markdown(
            "**既に1ファイル2-12秒程度の音声ファイル集とその書き起こしデータがある場合は、このタブは使用せずに学習できます。**"
        )
        
        gr.Markdown(
    """
    <p style='font-size: 20px; color: orange;'>⭐️と数字に沿って進みください。</p>
    """
    )
        

        # フィールドを使用して⭐1の部分を白色、サイズを元に戻します
        with gr.Row():
            with gr.Column():
                model_name = gr.Textbox(
                    label="⭐１.モデル名を入力してください（話者名としても使われます）⭐"
                )

        with gr.Accordion("音声のスライス"):
            gr.Markdown(
                "**すでに適度な長さの音声ファイルからなるデータがある場合は、その音声をData/{}/rawに入れれば、このステップは不要です。**"
            )
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="⭐２.音声の入っているフォルダパス⭐",
                        value="inputs",
                        info="下記フォルダにwavやmp3等のファイルを入れておいてください",
                    )
                    min_sec = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="この秒数未満は切り捨てる",
                    )
                    max_sec = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=12,
                        step=0.5,
                        label="この秒数以上は切り捨てる",
                    )
                    min_silence_dur_ms = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=700,
                        step=100,
                        label="無音とみなして区切る最小の無音の長さ（ms）",
                    )
                    time_suffix = gr.Checkbox(
                        value=False,
                        label="WAVファイル名の末尾に元ファイルの時間範囲を付与する",
                    )
                    slice_button = gr.Button("⭐３.スライスを実行⭐")
                result1 = gr.Textbox(label="結果")

        with gr.Row():
            with gr.Column():
                whisper_model = gr.Dropdown(
                    [
                        "tiny",
                        "base",
                        "small",
                        "medium",
                        "large",
                        "large-v2",
                        "large-v3",
                    ],
                    label="Whisperモデル",
                    value="large-v3",
                )
                use_hf_whisper = gr.Checkbox(
                    label="HuggingFaceのWhisperを使う（速度が速いがVRAMを多く使う）",
                    value=True,
                )
                hf_repo_id = gr.Dropdown(
                    ["openai/whisper", "kotoba-tech/kotoba-whisper-v1.1"],
                    label="HuggingFaceのWhisperモデル",
                    value="openai/whisper",
                )
                compute_type = gr.Dropdown(
                    [
                        "int8",
                        "int8_float32",
                        "int8_float16",
                        "int8_bfloat16",
                        "int16",
                        "float16",
                        "bfloat16",
                        "float32",
                    ],
                    label="計算精度",
                    value="bfloat16",
                    visible=False,
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    label="バッチサイズ",
                    info="大きくすると速度が速くなるがVRAMを多く使う",
                )
                language = gr.Dropdown(["ja", "en", "zh"], value="ja", label="言語")
                initial_prompt = gr.Textbox(
                    label="初期プロンプト",
                    value="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
                    info="このように書き起こしてほしいという例文（句読点の入れ方・笑い方・固有名詞等）",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="ビームサーチのビーム数",
                    info="小さいほど速度が上がる（以前は5）",
                )
            transcribe_button = gr.Button("⭐4.音声の文字起こし⭐")
            result2 = gr.Textbox(label="結果")

        slice_button.click(
            do_slice,
            inputs=[
                model_name,
                min_sec,
                max_sec,
                min_silence_dur_ms,
                time_suffix,
                input_dir,
            ],
            outputs=[result1],
        )

        transcribe_button.click(
            do_transcribe,
            inputs=[
                model_name,
                whisper_model,
                compute_type,
                language,
                initial_prompt,
                use_hf_whisper,
                batch_size,
                num_beams,
                hf_repo_id,
            ],
            outputs=[result2],
        )

        use_hf_whisper.change(
            lambda x: (
                gr.update(visible=x),
                gr.update(visible=x),
                gr.update(visible=not x),
            ),
            inputs=[use_hf_whisper],
            outputs=[hf_repo_id, batch_size, compute_type],
        )

    return app
