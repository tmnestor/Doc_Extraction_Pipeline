"""The vLLM engine, as the pipeline calls it.

Every request the pipeline makes of the model goes through ``VllmBackend.chat``: a list of images with
a prompt for each, sent to ``LLM.chat`` as one call so that the engine's scheduler can run up to
``max_num_seqs`` of them at once. A page on its own is a list of one.

vLLM prepares each image with the model's own Hugging Face processor and works out where the image's
placeholder tokens go in the prompt. The order of the image and the text in a request is the order the
model's chat template renders them in, so it is part of what the model is sent.
"""

import base64
import io
import logging
from typing import Any

from PIL import Image

from common import prompt_trace

logger = logging.getLogger(__name__)


class VllmBackend:
    """One vLLM engine and the way requests are built for it."""

    def __init__(
        self,
        engine: Any,
        *,
        model_type_key: str,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        default_image_first: bool = False,
        trace_path: str | None = None,
        debug: bool = False,
    ) -> None:
        self.model = engine
        self._model_type_key = model_type_key
        # A chat template to use in place of the model's own, or None for the model's own.
        self._chat_template = chat_template
        # Arguments for the chat template, from the model's registration: for a template that can think,
        # {"enable_thinking": False}.
        self._chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs else {}
        # Whether this model's own request form puts the image before the text.
        self._default_image_first = default_image_first
        self._debug = debug
        # How many prompt tokens the engine served from its cache, of how many, over this backend's calls.
        self._cached_prompt_tokens = 0
        self._total_prompt_tokens = 0
        self._cache_stats_available = True
        self._requests = 0
        if trace_path:
            prompt_trace.enable(trace_path)

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """A PIL image as a PNG in a base64 data URL."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def _conversation(self, image: Image.Image, prompt: str, *, image_first: bool) -> list[dict[str, Any]]:
        """One request as a conversation of one user message."""
        text = {"type": "text", "text": prompt}
        picture = {"type": "image_url", "image_url": {"url": self._encode_image(image)}}
        return [{"role": "user", "content": [picture, text] if image_first else [text, picture]}]

    def chat(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_tokens: int,
        *,
        text_first: bool = False,
    ) -> list[str]:
        """Send one request for each image, all in one engine call, and give the replies in order.

        Args:
            images: One image a request.
            prompts: The prompt of each request, in the same order.
            max_tokens: The completion budget of every request.
            text_first: The request form receipts, invoices and the SROIE benchmark have been read
                with: the text before the image, and no chat-template arguments. False is the model's
                own form: its own order, with the arguments its registration gives.

        Raises:
            ValueError: If there is not one prompt for each image.
        """
        if len(images) != len(prompts):
            raise ValueError(
                f"What: {len(images)} images were given with {len(prompts)} prompts.\n"
                f"Where: models/backends/vllm_backend.py, VllmBackend.chat.\n"
                f"Expected: one prompt for each image, in the same order.\n"
                f"How to fix: build both lists in one loop. Replies are matched to pages by position, so a "
                f"list that is short would score answers against the wrong pages."
            )
        if not images:
            return []
        from vllm import SamplingParams

        image_first = False if text_first else self._default_image_first
        conversations = [
            self._conversation(image, prompt, image_first=image_first)
            for image, prompt in zip(images, prompts, strict=True)
        ]
        template_arguments: dict[str, Any] = {}
        if self._chat_template_kwargs and not text_first:
            template_arguments["chat_template_kwargs"] = dict(self._chat_template_kwargs)
        outputs = self.model.chat(
            messages=conversations,
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
            chat_template=self._chat_template,
            use_tqdm=False,
            **template_arguments,
        )

        replies = []
        for output, prompt in zip(outputs, prompts, strict=True):
            self._count_cached_tokens(output)
            reply = output.outputs[0].text.strip()
            if prompt_trace.is_enabled():
                prompt_ids = getattr(output, "prompt_token_ids", None)
                reply_ids = output.outputs[0].token_ids
                prompt_trace.record(
                    prompt=prompt,
                    response=reply,
                    model=self._model_type_key,
                    prompt_tokens=len(prompt_ids) if prompt_ids else None,
                    completion_tokens=len(reply_ids) if reply_ids else None,
                )
            replies.append(reply)
        return replies

    def _count_cached_tokens(self, output: Any) -> None:
        """Add one reply's cached and total prompt tokens to the running counts.

        A build of vLLM whose replies lack either number gives none for the whole run, so the first
        reply without them marks the counts as unavailable.
        """
        self._requests += 1
        cached = getattr(output, "num_cached_tokens", None)
        prompt_ids = getattr(output, "prompt_token_ids", None)
        if cached is None or prompt_ids is None:
            self._cache_stats_available = False
            return
        self._cached_prompt_tokens += int(cached)
        self._total_prompt_tokens += len(prompt_ids)

    def cache_hit_summary(self) -> dict[str, Any]:
        """How much of the prompts the engine served from its cache, over this backend's calls."""
        if not self._cache_stats_available:
            return {"available": False, "calls": self._requests}
        total = self._total_prompt_tokens
        return {
            "available": True,
            "calls": self._requests,
            "cached_prompt_tokens": self._cached_prompt_tokens,
            "total_prompt_tokens": total,
            "hit_ratio": (self._cached_prompt_tokens / total) if total else 0.0,
        }
