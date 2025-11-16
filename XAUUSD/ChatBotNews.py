# gold_news_analyzer.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import pandas as pd

class GoldNewsAnalyzer:
    def __init__(self, google_api_key, max_memory=3):
        self.google_api_key = google_api_key
        self.memory_list = []
        self.max_memory = max_memory
        
        # สร้าง Prompt Template
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""
ต่อไปนี้คือการสนทนาระหว่างมนุษย์กับ AI ผู้ช่วย
AI มีหน้าที่วิเคราะห์ข้อมูลข่าวและผลการคาดการณ์จากโมเดล ML เพื่อประเมินว่าเนื้อหาของข่าวอาจมีแนวโน้มทำให้ราคาทองขึ้นหรือลง โดยจะให้น้ำหนักไปทางข่าวมากกว่าโมเดล ML

ข้อมูลข่าวล่าสุด:
{history}

ผลการคาดการจากโมเดล:
{input}

AI โปรดตอบโดยสรุปว่า ข่าวนี้อาจทำให้ราคาทองขึ้นหรือราคาทองลง พร้อมเหตุผลสั้น ๆ:
"""
        )
        
        # โหลดโมเดล Gemini ผ่าน LangChain
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.google_api_key,
            temperature=0.7,
            max_output_tokens=300
        )

    # ฟังก์ชันดึงข่าวจาก Mining.com
    @staticmethod
    def fetch_gold_news_mining(max_articles=10):
        url = "https://www.mining.com/commodity/gold/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/116.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"Error fetching page: status {resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.select("article.archive-article")

        results = []
        for art in articles[:max_articles]:
            date_tag = art.find("p", class_="date")
            date = date_tag.get_text(strip=True) if date_tag else None

            title_tag = art.find("h4", class_="title").find("a")
            title = title_tag.get_text(strip=True) if title_tag else None
            link = title_tag.get("href") if title_tag else None

            excerpt_tag = art.find("p", class_="excerpt").find("a")
            excerpt = excerpt_tag.get_text(strip=True) if excerpt_tag else None

            results.append({
                "date": date,
                "title": title,
                "url": link,
                "excerpt": excerpt
            })

        return pd.DataFrame(results)

    # ฟังก์ชันดึงบทความเต็ม
    @staticmethod
    def fetch_full_article(url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/116.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"Error fetching page: status {resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        content_div = soup.find("div", class_="content")
        if not content_div:
            return ""
        paragraphs = content_div.find_all("p")
        article_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
        return article_text

    # ฟังก์ชันวิเคราะห์แนวโน้มราคาทอง
    def analyze_news(self, latest_price, ml_prediction, context_text):
        # สร้าง prompt จาก template
        formatted_prompt = self.prompt.format(
            history=context_text,
            input=f"ทำนาย: {ml_prediction} ราคาทองล่าสุด: {latest_price}"
        )
        # เรียก LLM โดยตรง
        response = self.llm.invoke(formatted_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        # print(f"คุณ: {user_question}")
        print(f"AI: {response_text.strip()}")
        return response_text.strip()

    # ฟังก์ชันเก็บข้อมูลข่าว + วิเคราะห์ลง CSV
    @staticmethod
    def save_to_csv(news_text, latest_price, ml_prediction, ai_analysis, csv_file="Data/gold_analysis.csv"):
        df = pd.DataFrame([{
            "ข่าว": news_text,
            "ราคาทองล่าสุด": latest_price,
            "ผลโมเดล": ml_prediction,
            "AI วิเคราะห์": ai_analysis
        }])
        try:
            df_existing = pd.read_csv(csv_file)
            df_final = pd.concat([df_existing, df], ignore_index=True)
        except FileNotFoundError:
            df_final = df
        df_final.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"ข้อมูลถูกบันทึกลงไฟล์ {csv_file} เรียบร้อยแล้ว")
