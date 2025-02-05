import requests
import os

class TelegramService:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendDocument"

    def send_file(self, file_path: str, caption: str = "📂 File phụ đề mới"):
        """Gửi file đến channel Telegram"""
        try:
            with open(file_path, "rb") as file:
                response = requests.post(
                    self.api_url,
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"document": file}
                )

            if response.status_code == 200:
                print(f"✅ Đã gửi file {file_path} đến Telegram!")
            else:
                print(f"❌ Lỗi gửi file: {response.text}")

        except Exception as e:
            print(f"⚠️ Lỗi khi gửi file: {str(e)}")
