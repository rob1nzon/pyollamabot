import asyncio
import logging
import os
from typing import Optional
from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.client import AsyncTcpClient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv(key="OLLAMA_HOST")
class WyomingTranscriber:
    def __init__(self, host: str = OLLAMA_HOST, port: int = 10300):
        self.host = host
        self.port = port
        self._client = None

    async def _ensure_connected(self) -> AsyncTcpClient:
        """Ensure connection to Wyoming service with retries."""
        if self._client and self._client.is_connected:
            return self._client

        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Connecting to Wyoming service at {self.host}:{self.port} (attempt {attempt + 1}/{max_retries})")
                self._client = AsyncTcpClient(self.host, self.port)
                await self._client.connect()
                logger.debug("Connected successfully")
                return self._client
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s")
                await asyncio.sleep(retry_delay)
        
        raise ConnectionError("Failed to connect after all retries")

    async def transcribe(
        self, 
        audio_data: bytes, 
        language: Optional[str] = "ru",
        initial_prompt: Optional[str] = None,
        timeout: float = 30.0
    ) -> Optional[str]:
        """
        Transcribe audio data using Wyoming protocol.
        
        Args:
            audio_data: Raw audio bytes (16kHz, 16-bit, mono)
            language: Optional language code (e.g. "en", "es", "fr")
            initial_prompt: Optional text to provide context for transcription
            timeout: Maximum time to wait for transcription in seconds
            
        Returns:
            Transcribed text or None if transcription failed
        """
        try:
            client = await self._ensure_connected()

            # Send transcribe event with language if specified
            if language or initial_prompt:
                logger.debug(f"Sending transcribe request: language={language}, initial_prompt={initial_prompt}")
                await client.write_event(Transcribe(
                    language=language,
                ).event())

            # Send audio format info
            logger.debug("Sending audio format info")
            await client.write_event(AudioStart(
                     16000,
                     2,
                    1
            ).event())

            # Send audio in chunks
            chunk_size = 1024 * 16  # 16KB chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await client.write_event(AudioChunk(16000,  2,  1,audio=chunk).event())
                logger.debug(f"Sent audio chunk {i//chunk_size + 1}, size: {len(chunk)} bytes")

            # Send stop message
            logger.debug("Sending stop message")
            await client.write_event(AudioStop().event())

            # Read responses with timeout
            logger.debug("Waiting for responses")

            while True:
                    event = await client.read_event()
                    if event is None:
                        logger.debug("Connection lost")
                        return None
                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        return transcript.text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None
            
        finally:
            # Don't disconnect - keep connection alive for future requests
            pass
