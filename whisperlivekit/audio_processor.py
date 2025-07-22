import asyncio
import numpy as np
from time import time, sleep
import math
import logging
import traceback
from datetime import timedelta
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper_streaming_custom.whisper_online import online_factory
from whisperlivekit.core import TranscriptionEngine

# Set up logging once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object() # unique sentinel object for end of stream marker

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

class AudioProcessor:
    """
    Processes audio streams for transcription.
    Handles audio processing, state management, and result formatting.
    """
    
    def __init__(self, **kwargs):
        """Initialize the audio processor with configuration, models, and state."""
        
        if 'transcription_engine' in kwargs and isinstance(kwargs['transcription_engine'], TranscriptionEngine):
            models = kwargs['transcription_engine']
        else:
            models = TranscriptionEngine(**kwargs)
        
        # Audio processing settings
        self.args = models.args
        self.sample_rate = 16000
        self.channels = 1
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.sample_rate * self.bytes_per_sample  # 16000 * 2 = 32000 bytes/sec
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)  # samples per chunk
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz

        # State management
        self.is_stopping = False
        self.tokens = []
        self.buffer_transcription = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.total_audio_duration = 0.0  # Track total audio duration sent
        self.stop_time = None  # Time when stopping was initiated
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""
        
        # Models and processing
        self.asr = models.asr
        self.tokenizer = models.tokenizer
        
        self.transcription_queue = asyncio.Queue() if self.args.transcription else None
        self.pcm_buffer = bytearray()

        # Task references
        self.transcription_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []
        
        # Initialize transcription engine if enabled
        if self.args.transcription:
            self.online = online_factory(self.args, models.asr, models.tokenizer)

    def convert_pcm_to_float(self, pcm_buffer):
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        """Thread-safe update of transcription with new data."""
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep

    async def get_current_state(self):
        """Get the current state of processing."""
        async with self.lock:
            return {
                "tokens": list(self.tokens),
                "buffer_transcription": self.buffer_transcription,
                "end_buffer": self.end_buffer,
                "full_transcription": self.full_transcription,
                "sep": self.sep,
            }

    async def transcription_processor(self):
        """Process audio chunks for transcription."""
        self.full_transcription = ""
        self.sep = self.online.asr.sep
        cumulative_pcm_duration_stream_time = 0.0
        
        while True:
            try:
                pcm_array = await self.transcription_queue.get()
                if pcm_array is SENTINEL:
                    logger.debug("Transcription processor received sentinel. Finishing.")
                    self.transcription_queue.task_done()
                    break
                
                if not self.online: # Should not happen if queue is used
                    logger.warning("Transcription processor: self.online not initialized.")
                    self.transcription_queue.task_done()
                    continue

                asr_internal_buffer_duration_s = len(getattr(self.online, 'audio_buffer', [])) / self.online.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.beg_loop - self.end_buffer)

                logger.info(
                    f"ASR processing: internal_buffer={asr_internal_buffer_duration_s:.2f}s, "
                    f"lag={transcription_lag_s:.2f}s."
                )
                
                # Process transcription
                duration_this_chunk = len(pcm_array) / self.sample_rate if isinstance(pcm_array, np.ndarray) else 0
                cumulative_pcm_duration_stream_time += duration_this_chunk
                stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time

                self.online.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)
                new_tokens, current_audio_processed_upto = self.online.process_iter()
                
                # Get buffer information
                _buffer_transcript_obj = self.online.get_buffer()
                buffer_text = _buffer_transcript_obj.text

                if new_tokens:
                    validated_text = self.sep.join([t.text for t in new_tokens])
                    self.full_transcription += validated_text
                    
                    if buffer_text.startswith(validated_text):
                        buffer_text = buffer_text[len(validated_text):].lstrip()

                candidate_end_times = [self.end_buffer]

                if new_tokens:
                    candidate_end_times.append(new_tokens[-1].end)
                
                if _buffer_transcript_obj.end is not None:
                    candidate_end_times.append(_buffer_transcript_obj.end)
                
                candidate_end_times.append(current_audio_processed_upto)
                
                new_end_buffer = max(candidate_end_times)
                
                await self.update_transcription(
                    new_tokens, buffer_text, new_end_buffer, self.full_transcription, self.sep
                )
                self.transcription_queue.task_done()
                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL : # Check if pcm_array was assigned from queue
                    self.transcription_queue.task_done()
        logger.info("Transcription processor task finished.")

    async def results_formatter(self):
        """Format processing results for output."""
        while True:
            try:
                # Get current state
                state = await self.get_current_state()
                tokens = state["tokens"]
                buffer_transcription = state["buffer_transcription"]
                sep = state["sep"]
                
                # Format output - simple transcription lines
                lines = []
                if tokens:
                    # Group consecutive tokens into lines
                    current_line_tokens = []
                    
                    for token in tokens:
                        if not token.is_dummy:
                            current_line_tokens.append(token)
                    
                    if current_line_tokens:
                        line = {
                            "start": current_line_tokens[0].start,
                            "end": current_line_tokens[-1].end,
                            "text": sep.join([t.text for t in current_line_tokens]).strip()
                        }
                        if line["text"]:
                            lines.append(line)
                
                # Calculate remaining processing time
                if self.is_stopping:
                    # When stopping, calculate remaining time based on total audio sent vs processed
                    remaining_time_transcription = max(0, self.total_audio_duration - self.end_buffer)
                    logger.debug(f"Stopping: total_audio={self.total_audio_duration:.2f}s, processed={self.end_buffer:.2f}s, remaining={remaining_time_transcription:.2f}s")
                else:
                    # When actively receiving audio, use current time
                    remaining_time_transcription = max(0, time() - self.beg_loop - self.end_buffer)
                
                # Format for JSON output
                response_content = {
                    "status": "processing" if not self.is_stopping else "stopping",
                    "lines": lines,
                    "buffer_transcription": buffer_transcription,
                    "remaining_time_transcription": remaining_time_transcription
                }
                
                # Only yield if content has changed or it's been a while
                current_content = str(response_content)
                if current_content != self.last_response_content or remaining_time_transcription < 0.001:
                    yield response_content
                    self.last_response_content = current_content
                
                # Stop if processing is complete (only check transcription)
                if self.is_stopping and remaining_time_transcription <= 0.001:
                    logger.info("Transcription processing complete. Results formatter finishing.")
                    break
                    
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(1)
        
    async def create_tasks(self):
        """Create and start processing tasks."""
        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog = []

        if self.args.transcription and self.online:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)

        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)
        
        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        while True:
            try:
                await asyncio.sleep(10)
                
                for i, task in enumerate(tasks_to_monitor):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally.")
                            
            except asyncio.CancelledError:
                logger.info("Watchdog task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in watchdog task: {e}", exc_info=True)
        
    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")        
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()
        
        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")
        
        logger.info("AudioProcessor cleanup complete.")

    async def process_audio(self, message):
        """Process incoming PCM audio data directly."""
        if not message:
            logger.info("Empty audio message received, initiating stop sequence.")
            self.is_stopping = True
            self.stop_time = time()
            
            # Process any remaining audio in the buffer before stopping
            if len(self.pcm_buffer) > 0:
                logger.info(f"Processing final {len(self.pcm_buffer)} bytes of buffered audio")
                pcm_array = self.convert_pcm_to_float(self.pcm_buffer)
                self.pcm_buffer.clear()
                
                # Send final chunk to transcription if enabled
                if self.args.transcription and self.transcription_queue:
                    await self.transcription_queue.put(pcm_array.copy())
            
            # Signal end of stream to transcription processor
            if self.args.transcription and self.transcription_queue:
                await self.transcription_queue.put(SENTINEL)
            return

        if self.is_stopping:
            logger.warning("AudioProcessor is stopping. Ignoring incoming audio.")
            return

        try:
            # Assume message is already PCM s16le format
            self.pcm_buffer.extend(message)
            
            # Track total audio duration sent
            audio_duration = len(message) / self.bytes_per_sec
            self.total_audio_duration += audio_duration

            # Process when enough data is accumulated
            if len(self.pcm_buffer) >= self.bytes_per_sec:
                if len(self.pcm_buffer) > self.max_bytes_per_sec:
                    logger.warning(
                        f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                        f"Consider using a smaller model."
                    )

                # Process audio chunk
                pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:self.max_bytes_per_sec])
                self.pcm_buffer = self.pcm_buffer[self.max_bytes_per_sec:]
                
                # Send to transcription if enabled
                if self.args.transcription and self.transcription_queue:
                    await self.transcription_queue.put(pcm_array.copy())

        except Exception as e:
            logger.error(f"Error processing PCM audio data: {e}")
