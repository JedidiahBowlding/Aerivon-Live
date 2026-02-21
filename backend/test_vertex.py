from google import genai
import os

client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with OK"
)

print(resp.text)
