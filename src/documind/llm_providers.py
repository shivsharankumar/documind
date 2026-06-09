from google import genai
from groq import AsyncGroq

from documind.config import settings


class GroqProvider:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    async def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


class GeminiProvider:
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    async def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        # Gemini's API mixes system into the prompt differently
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=f"{system}\n\nUser: {user}",
            config={"temperature": temperature, "max_output_tokens": 1024},
        )
        return resp.text or ""
