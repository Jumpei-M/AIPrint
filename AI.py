import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image
import torch
import threading
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator

# --- グローバル変数 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None
translator = None
model_ready = False

# --- モデル読み込み関数（スレッド用） ---
def load_model_async(load_status, start_button):
    global processor, model, translator, model_ready
    try:
        load_status.set("🔄 モデルを読み込み中...")

        # 実行ファイルのあるディレクトリを基準にモデルフォルダを探す
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base_dir, "models", "blip")

        # モデルをローカルフォルダから読み込み
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=cache_dir
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=cache_dir
        ).to(device)

        translator = Translator()
        model_ready = True
        load_status.set("✅ モデル読み込み完了")
        start_button.config(state="normal")
    except Exception as e:
        load_status.set("❌ モデル読み込み失敗")
        messagebox.showerror("エラー", f"モデルの読み込みに失敗しました:\n{e}")

# --- 画像解析関数 ---
def analyze_image(file_path, progress_var, result_label):
    try:
        progress_var.set(10)
        image = Image.open(file_path).convert('RGB')
        progress_var.set(30)

        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        progress_var.set(60)

        translated_caption = translator.translate(caption, src='en', dest='ja').text
        progress_var.set(80)

        enriched_caption = f"これは、「{translated_caption}」の様子です。"
        progress_var.set(100)

        result_text = f"【解析結果（英語）】\n{caption}\n\n【日本語訳】\n{enriched_caption}"
        result_label.config(text=result_text)
    except Exception as e:
        messagebox.showerror("エラー", f"処理中に問題が発生しました:\n{e}")

# --- GUI構築 ---
def run_gui():
    def start_analysis():
        if not model_ready:
            messagebox.showinfo("準備中", "モデルの読み込みが完了していません。しばらくお待ちください。")
            return

        file_path = filedialog.askopenfilename(
            title="画像ファイルを選択",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.jfif")]
        )
        if not file_path:
            return

        result_label.config(text="解析中...")
        progress_var.set(0)
        threading.Thread(target=analyze_image, args=(file_path, progress_var, result_label)).start()

    root = tk.Tk()
    root.title("画像認識 × 翻訳付きAI")
    root.geometry("600x420")

    load_status = tk.StringVar(value="🔄 モデル準備中...")
    tk.Label(root, textvariable=load_status, font=("Arial", 10), fg="blue").pack()

    start_button = tk.Button(root, text="画像を選択して解析開始", command=start_analysis,
                             font=("Arial", 14), state="disabled")
    start_button.pack(pady=20)

    progress_var = tk.IntVar()
    ttk.Progressbar(root, maximum=100, variable=progress_var, length=400).pack(pady=10)

    result_label = tk.Label(root, text="", wraplength=550, justify="left", font=("Arial", 12))
    result_label.pack(pady=20)

    # 非同期でモデルロード（キャッシュ有効）
    threading.Thread(target=load_model_async, args=(load_status, start_button)).start()
    root.mainloop()

# --- 実行 ---
if __name__ == "__main__":
    run_gui()
