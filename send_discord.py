import requests

def send_discord_webhook(webhook_url, message, image_path=None, file_path=None):
    """
    ส่งข้อความ, รูป, และไฟล์ CSV ไปที่ Discord
    """
    data = {"content": message}
    files = {}

    if image_path:
        files["file1"] = open(image_path, "rb")
    if file_path:
        files["file2"] = open(file_path, "rb")

    response = requests.post(webhook_url, data=data, files=files if files else None)
    if response.status_code == 204 or response.status_code == 200:
        print("✅ ส่งข้อมูลไป Discord เรียบร้อย")
    else:
        print(f"❌ ล้มเหลว: {response.status_code}, {response.text}")

    # ปิดไฟล์
    for f in files.values():
        f.close()
