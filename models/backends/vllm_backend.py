"""The vLLM engine, as the pipeline calls it.

Every request the pipeline makes of the model goes through ``VllmBackend.chat``: a list of images with
a prompt for each, sent to ``LLM.chat`` as one call so that the engine's scheduler can run up to
``max_num_seqs`` of them at once. A page on its own is a list of one.

An image is handed over as a PIL image (the ``image_pil`` content part), not encoded into a URL for
vLLM to decode again. vLLM prepares it with the model's own Hugging Face processor and works out where
the image's placeholder tokens go in the prompt. The order of the image and the text in a request is the order the
model's chat template renders them in, so it is part of what the model is sent. Every request has the
one form: the order the model's registration gives (for both Gemma 4 models the text first), with the
chat-template arguments the registration gives.
"""

import logging
from pathlib import Path
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
        reply_tokens_added: int = 0,
        trace_path: Path | None = None,
    ) -> None:
        self.model = engine
        # Tokens added to the budget of every reply, from the model's registration: 0 for a model that
        # answers at once.
        self._reply_tokens_added = reply_tokens_added
        self._model_type_key = model_type_key
        # A chat template to use in place of the model's own, or None for the model's own.
        self._chat_template = chat_template
        # Arguments for the chat template, from the model's registration: for a template that can think,
        # {"enable_thinking": False}.
        self._chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs else {}
        # Whether this model's own request form puts the image before the text.
        self._default_image_first = default_image_first
        if trace_path:
            prompt_trace.enable(trace_path)

    @staticmethod
    def _as_rgb(image: Image.Image) -> Image.Image:
        """The image in RGB, which is what the model's processor works on.

        An RGB image is returned as it is. One with transparency is put on a white background first, as
        vLLM does with an image it loads itself: dropping the transparency alone would leave whatever
        colour lay under it, often black.
        """
        if image.mode == "RGB":
            return image
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            layered = image.convert("RGBA")
            background = Image.new("RGBA", layered.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, layered).convert("RGB")
        return image.convert("RGB")

    def _conversation(self, image: Image.Image, prompt: str, *, image_first: bool) -> list[dict[str, Any]]:
        """One request as a conversation of one user message."""
        text = {"type": "text", "text": prompt}
        picture = {"type": "image_pil", "image_pil": self._as_rgb(image)}
        return [{"role": "user", "content": [picture, text] if image_first else [text, picture]}]

    def chat(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_tokens: int,
    ) -> list[str]:
        """Send one request for each image, all in one engine call, and give the replies in order.

        Args:
            images: One image a request.
            prompts: The prompt of each request, in the same order.
            max_tokens: The completion budget of every request.

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

        conversations = [
            self._conversation(image, prompt, image_first=self._default_image_first)
            for image, prompt in zip(images, prompts, strict=True)
        ]
        template_arguments: dict[str, Any] = {}
        if self._chat_template_kwargs:
            template_arguments["chat_template_kwargs"] = dict(self._chat_template_kwargs)
        outputs = self.model.chat(
            messages=conversations,
            sampling_params=SamplingParams(max_tokens=max_tokens + self._reply_tokens_added, temperature=0),
            chat_template=self._chat_template,
            use_tqdm=False,
            **template_arguments,
        )

        replies = []
        for output, prompt in zip(outputs, prompts, strict=True):
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
