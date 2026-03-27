from google import genai
from google.genai import types
import os

from app.services.llm.base import BaseLLM
from app.core.config import settings


class GeminiLLM(BaseLLM):
    def __init__(self):
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)

    def enhance_description(self, image_paths: str, item_name: str, description: str = None) -> str:
        # prompt = f"""
        #     The user is tracking an item: "{item_name}"

        #     {f'and user description: "{description}"' if description else ""}

        #     You are given multiple images of the item. 

        #     Focus ONLY on this item in the image.
        #     Ignore other objects.

        #     Describe:
        #     - Give small description of the item.
        #     - Focus on where it is placed
        #     - and nearby context,
        #     - Try to analyse the room if you think its inside a room (like living room, bedroom, kitchen etc) and give details about that.
        #     - Important: combine information across images to give a more complete description.

        #     This is need because use might forget where they placed the item and will rely on this description to find it.
        #     Use the image to give as much details as possible about the item's location and surroundings.
        #     Be precise.

        #     IMPORTANT 1: If you are not sure about something, do not guess. Just skip it. It's better to give less information than wrong information.
        #     IMPORTANT 2: Use simple language to describe the location, so that it's easy for user to understand and find the item.
        #     """
        prompt = f"""
            You are an intelligent assistant helping users remember where they placed their personal items.

            DO NOT return any text before or after JSON.
            DO NOT include words like DESCRIPTION or explanation.

            CONTEXT:
            Users often forget where they keep things like remotes, chargers, or keys.
            They will rely on your response later to physically locate the item in their home.
            The user will physically walk to find the item based on your description.
            So your description must be:
            - clear
            - precise
            - easy to follow
            - based on stable visual cues

            The user is tracking: "{item_name}"
            {f'User description: "{description}"' if description else ""}

            You are given multiple images:
            - some show the item
            - some show the surroundings

            TASK:

            1. Generate a SHORT human-friendly description:
            - Help the user FIND the item quickly
            - Use directions like "next to", "in front of", "on top of"
            - Mention nearby stable objects (table, bed, wardrobe) with little details about them (color, position in the room, design) to help user visually locate the item
            - Keep it concise (2-3 lines max)

            2. Extract structured JSON:

            Format EXACTLY:

            DESCRIPTION:
            <short helpful description>

            Return ONLY valid JSON in this format:

            {{
            "description": "...",
            "confidence": 0.0,
            "structured": {{
                "object": "...",
                "color": "...",
                "room": "...",
                "surface": "...",
                "relative_position": "...",
                "nearby_objects": ["...", "..."],
                "room_landmarks": ["...", "..."]
            }}
            }}

            Confidence rules:
            - Never return 1.0
            - Use:
            0.8–0.9 → clear view
            0.6–0.8 → some uncertainty
            <0.6 → unclear

            RULES:
            - Output MUST start with {{ and end with }}
            - confidence must be a number between 0 and 1
            - base it on how clearly the item and location are visible
            - Focus ONLY on the given item
            - Do NOT guess if unsure
            - Prefer stable objects (bed, wardrobe, table) over small objects
            - Keep JSON values short and simple
            - description must be 2-3 short sentences
            - No markdown
            - No extra text
            """

        contents = [prompt]
        for path in image_paths:
             with open(path, "rb") as f:
                image_bytes = f.read()

             contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                )
            )

        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents
        )

        return response.text.strip()