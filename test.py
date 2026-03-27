from google import genai
import os

client = genai.Client(api_key="AIzaSyDoYEepU9UAvwi3XUA2RANQZ-CpNH7u4vc")

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Say hello"
)

print(response.text)