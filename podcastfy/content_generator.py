"""
Content Generator Module

This module is responsible for generating Q&A content based on input texts using
LangChain and various LLM backends. It handles the interaction with the AI model and
provides methods to generate and save the generated content.
"""

import logging
import os
import re
from abc import ABC, abstractmethod

from langchain import hub
from langchain.llms.base import BaseLLM
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from podcastfy.llm.factory import LLMProviderFactory, detect_llm_provider
from podcastfy.utils.config import ContentGeneratorConfigModel, load_app_config_model
from podcastfy.utils.config_conversation import ConversationConfigModel, load_conversation_config_model
from podcastfy.utils.constants import (
    COMMON_SSML_TAGS,
    DEFAULT_GEMINI_LLM,
    LOCAL_LLM_PLACEHOLDER,
)
from podcastfy.utils.enums import ApiKeyLabel, LLMProvider, SpeakerTag

logger = logging.getLogger(__name__)

_GREETING_TEMPLATE = (
    "ALWAYS START THE CONVERSATION GREETING THE AUDIENCE: Welcome to {podcast_name} - {podcast_tagline}."
)
_DEFAULT_CONVERSATION_PROMPT = "Please analyze this input and generate a conversation. {input_text}"
_DEFAULT_IMAGE_DETAIL: str = "high"


class PromptParams(BaseModel):
    input_text: str = ""
    conversation_style: list[str] = Field(default_factory=list)
    roles_person1: str = ""
    roles_person2: str = ""
    dialogue_structure: list[str] = Field(default_factory=list)
    podcast_name: str = ""
    podcast_tagline: str = ""
    output_language: str = "English"
    user_instructions: str = ""
    engagement_techniques: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class LLMBackend:
    def __init__(
        self,
        is_local: bool,
        temperature: float,
        max_output_tokens: int,
        model_name: str,
        api_key_label: str = ApiKeyLabel.GEMINI,
    ):
        self.is_local = is_local
        self.model_name = model_name
        self.is_multimodal = not is_local

        if is_local:
            provider_name = LLMProvider.LLAMAFILE
            api_key = ""
        else:
            provider_name = detect_llm_provider(model_name)
            api_key = os.environ.get(ApiKeyLabel.GEMINI if provider_name == LLMProvider.GEMINI else api_key_label, "")

        self.provider = LLMProviderFactory.create(
            provider_name,
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_key_label=api_key_label,
        )
        self.llm = self.provider.llm


class LongFormContentGenerator:
    """
    Handles generation of long-form podcast conversations by breaking content into manageable chunks.

    Uses a "Content Chunking with Contextual Linking" strategy to maintain context between segments
    while generating longer conversations.

    Attributes:
        LONGFORM_INSTRUCTIONS (str): Constant containing instructions for long-form generation
        llm_chain: The LangChain chain used for content generation
    """

    # Add constant for long-form instructions
    LONGFORM_INSTRUCTIONS = """
    Additional Instructions:
        1. Provide extensive examples and real-world applications
        2. Include detailed analysis and multiple perspectives
        3. Use the "yes, and" technique to build upon points
        4. Incorporate relevant anecdotes and case studies
        5. Balance detailed explanations with engaging dialogue
        6. Maintain consistent voice throughout the extended discussion
        7. Generate a long conversation - output max_output_tokens tokens
    """

    def __init__(self, chain: Runnable, llm: BaseLLM, config_conversation: ConversationConfigModel):
        """
        Initialize ConversationGenerator.

        Args:
            llm_chain: The LangChain chain to use for generation
            config_conversation: Conversation configuration dictionary
        """
        self.llm_chain = chain
        self.llm = llm
        self.max_num_chunks = config_conversation.max_num_chunks
        self.min_chunk_size = config_conversation.min_chunk_size

    def __calculate_chunk_size(self, input_content: str) -> int:
        """
        Calculate chunk size based on input content length.

        Args:
            input_content: Input text content

        Returns:
            Calculated chunk size that ensures:
            - Returns 1 if content length <= min_chunk_size
            - Each chunk has at least min_chunk_size characters
            - Number of chunks is at most max_num_chunks
        """
        input_length = len(input_content)
        if input_length <= self.min_chunk_size:
            return input_length

        maximum_chunk_size = input_length // self.max_num_chunks
        if maximum_chunk_size >= self.min_chunk_size:
            return maximum_chunk_size

        # Calculate chunk size that maximizes size while maintaining minimum chunks
        return input_length // (input_length // self.min_chunk_size)

    def chunk_content(self, input_content: str, chunk_size: int) -> list[str]:
        """
        Split input content into manageable chunks while preserving context.

        Args:
            input_content (str): The input text to chunk
            chunk_size (int): Maximum size of each chunk

        Returns:
            List[str]: List of content chunks
        """
        sentences = input_content.split(". ")
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
        return chunks

    def enhance_prompt_params(self, prompt_params: dict, part_idx: int, total_parts: int, chat_context: str) -> dict:
        """
        Enhance prompt parameters for long-form content generation.

        Args:
            prompt_params (Dict): Original prompt parameters
            part_idx (int): Index of current conversation part
            total_parts (int): Total number of conversation parts
            chat_context (str): Chat context from previous parts

        Returns:
            Dict: Enhanced prompt parameters with part-specific instructions
        """
        enhanced_params = prompt_params.copy()
        # Initialize part_instructions with chat context
        enhanced_params["context"] = chat_context

        COMMON_INSTRUCTIONS = """
            Podcast conversation so far is given in CONTEXT.
            Continue the natural flow of conversation. Follow-up on the very previous point/question without repeating topics or points already discussed!
            Hence, the transition should be smooth and natural. Avoid abrupt transitions.
            Make sure the first to speak is different from the previous speaker. Look at the last tag in CONTEXT to determine the previous speaker.
            If last tag in CONTEXT is <Person1>, then the first to speak now should be <Person2>.
            If last tag in CONTEXT is <Person2>, then the first to speak now should be <Person1>.
            This is a live conversation without any breaks.
            Hence, avoid statemeents such as "we'll discuss after a short break.  Stay tuned" or "Okay, so, picking up where we left off".
        """

        # Add part-specific instructions
        if part_idx == 0:
            enhanced_params["instruction"] = f"""
{_GREETING_TEMPLATE.format(podcast_name=enhanced_params["podcast_name"], podcast_tagline=enhanced_params["podcast_tagline"])}
You are generating the Introduction part of a long podcast conversation.
            Don't cover any topics yet, just introduce yourself and the topic. Leave the rest for later parts, following these guidelines:
            """
        elif part_idx == total_parts - 1:
            enhanced_params["instruction"] = f"""
            You are generating the last part of a long podcast conversation.
            {COMMON_INSTRUCTIONS}
            For this part, discuss the below INPUT and then make concluding remarks in a podcast conversation format and END THE CONVERSATION GREETING THE AUDIENCE WITH PERSON1 ALSO SAYING A GOOD BYE MESSAGE, following these guidelines:
            """
        else:
            enhanced_params["instruction"] = f"""
            You are generating part {part_idx + 1} of {total_parts} parts of a long podcast conversation.
            {COMMON_INSTRUCTIONS}
            For this part, discuss the below INPUT in a podcast conversation format, following these guidelines:
            """

        return enhanced_params

    def generate_long_form(self, input_content: str, prompt_params: dict) -> str:
        """
        Generate a complete long-form conversation using chunked content.

        Args:
            input_content (str): Input text for conversation
            prompt_params (dict): Base prompt parameters

        Returns:
            str: Generated long-form conversation
        """
        # Add long-form instructions once at the beginning
        prompt_params["user_instructions"] = prompt_params.get("user_instructions", "") + self.LONGFORM_INSTRUCTIONS

        # Get chunk size
        chunk_size = self.__calculate_chunk_size(input_content)

        chunks = self.chunk_content(input_content, chunk_size)
        conversation_parts = []
        chat_context = input_content
        num_parts = len(chunks)

        for i, chunk in enumerate(chunks):
            enhanced_params = self.enhance_prompt_params(
                prompt_params, part_idx=i, total_parts=num_parts, chat_context=chat_context
            )
            enhanced_params["input_text"] = chunk
            response = self.llm_chain.invoke(enhanced_params)
            chat_context = response if i == 0 else chat_context + response
            conversation_parts.append(response)

        return self.stitch_conversations(conversation_parts)

    def stitch_conversations(self, parts: list[str]) -> str:
        """
        Combine conversation parts with smooth transitions.

        Args:
            parts (List[str]): List of conversation parts

        Returns:
            str: Combined conversation
        """
        # Simply join the parts, preserving all markup
        return "\n".join(parts)


# Make BaseContentCleaner a mixin class
class ContentCleanerMixin:
    """
    Mixin class containing common transcript cleaning operations.

    Provides reusable cleaning methods that can be used by different content generation strategies.
    Methods use protected naming convention (_method_name) as they are intended for internal use
    by the strategies.
    """

    @staticmethod
    def _clean_scratchpad(text: str) -> str:
        """
        Remove scratchpad blocks, plaintext blocks, standalone triple backticks, any string enclosed in brackets, and underscores around words.
        """
        try:
            pattern = r"```scratchpad\n.*?```\n?|```plaintext\n.*?```\n?|```\n?|\[.*?\]"
            cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
            # Remove "xml" if followed by </Person1> or </Person2>
            cleaned_text = re.sub(r"xml(?=\s*</Person[12]>)", "", cleaned_text)
            # Remove underscores around words
            cleaned_text = re.sub(r"_(.*?)_", r"\1", cleaned_text)
            return cleaned_text.strip()
        except Exception as e:
            logger.error(f"Error cleaning scratchpad content: {str(e)}")
            # Best-effort fallback: return original text on any error
            return text

    @staticmethod
    def _clean_tss_markup(
        input_text: str, additional_tags: list[SpeakerTag] | None = None
    ) -> str:
        if additional_tags is None:
            additional_tags = [SpeakerTag.PERSON1, SpeakerTag.PERSON2]
        try:
            from podcastfy.tts.base import TTSProvider

            input_text = ContentCleanerMixin._clean_scratchpad(input_text)
            supported_tags = [*COMMON_SSML_TAGS.copy(), "speak"]
            tts_provider = TTSProvider.__new__(TTSProvider)
            return tts_provider.clean_tss_markup(
                input_text,
                additional_tags=list(additional_tags),
                supported_tags=supported_tags,
            )
        except Exception as e:
            logger.error(f"Error cleaning TSS markup: {str(e)}")
            # Best-effort fallback: return original input on any error
            return input_text


class ContentGenerationStrategy(ABC):
    @abstractmethod
    def validate(self, input_texts: str, image_file_paths: list[str]) -> None:
        pass

    @abstractmethod
    def generate(self, chain, input_texts: str, prompt_params: PromptParams, **kwargs) -> str:
        pass

    @abstractmethod
    def clean(self, response: str, config: ContentGeneratorConfigModel) -> str:
        pass

    @abstractmethod
    def compose_prompt_params(
        self,
        config_conversation: ConversationConfigModel,
        image_file_paths: list[str] | None = None,
        image_path_keys: list[str] | None = None,
        input_texts: str = "",
    ) -> PromptParams:
        pass


class StandardContentStrategy(ContentGenerationStrategy, ContentCleanerMixin):
    def __init__(
        self, llm, content_generator_config: ContentGeneratorConfigModel, config_conversation: ConversationConfigModel
    ):
        self.llm = llm
        self.content_generator_config = content_generator_config
        self.config_conversation = config_conversation

    def validate(self, input_texts: str, image_file_paths: list[str]) -> None:
        pass

    def generate(self, chain, input_texts: str, prompt_params: PromptParams, **kwargs) -> str:
        return chain.invoke(prompt_params)

    def clean(self, response: str, config: ContentGeneratorConfigModel) -> str:
        return self._clean_tss_markup(response)

    def compose_prompt_params(
        self,
        config_conversation: ConversationConfigModel,
        image_file_paths: list[str] | None = None,
        image_path_keys: list[str] | None = None,
        input_texts: str = "",
    ) -> PromptParams:
        if image_file_paths is None:
            image_file_paths = []
        if image_path_keys is None:
            image_path_keys = []
        return PromptParams(
            input_text=input_texts,
            conversation_style=config_conversation.conversation_style,
            roles_person1=config_conversation.roles_person1,
            roles_person2=config_conversation.roles_person2,
            dialogue_structure=config_conversation.dialogue_structure,
            podcast_name=config_conversation.podcast_name,
            podcast_tagline=config_conversation.podcast_tagline,
            output_language=config_conversation.output_language,
            engagement_techniques=config_conversation.engagement_techniques,
        )



class LongFormContentStrategy(ContentGenerationStrategy, ContentCleanerMixin):
    def __init__(
        self, llm, content_generator_config: ContentGeneratorConfigModel, config_conversation: ConversationConfigModel
    ):
        self.llm = llm
        self.content_generator_config = content_generator_config
        self.config_conversation = config_conversation

    def validate(self, input_texts: str, image_file_paths: list[str]) -> None:
        if not input_texts.strip():
            msg = "Long-form generation requires non-empty input text"
            raise ValueError(msg)
        if image_file_paths:
            msg = "Long-form generation is not available with image inputs"
            raise ValueError(msg)

    def generate(self, chain, input_texts: str, prompt_params: PromptParams, **kwargs) -> str:
        generator = LongFormContentGenerator(chain, self.llm, self.config_conversation)
        return generator.generate_long_form(input_texts, prompt_params)

    def clean(self, response: str, config: ContentGeneratorConfigModel) -> str:
        standard_clean = self._clean_tss_markup(response)
        return self._clean_transcript_response(standard_clean, config)

    def _clean_transcript_response(self, transcript: str, config: ContentGeneratorConfigModel) -> str:
        logger.debug("Starting transcript cleaning process")

        final_transcript = self._fix_alternating_tags(transcript)

        logger.debug("Completed transcript cleaning process")

        return final_transcript

    def _fix_alternating_tags(self, transcript: str) -> str:
        try:
            pattern = r"(<Person[12]>.*?</Person[12]>)"
            blocks = re.split(pattern, transcript, flags=re.DOTALL)

            blocks = [b.strip() for b in blocks if b.strip()]

            merged_blocks = []
            current_content = []
            current_person = None

            for block in blocks:
                match = re.match(r"<Person([12])>(.*?)</Person\1>", block, re.DOTALL)
                if not match:
                    continue

                person_num, content = match.groups()
                content = content.strip()

                if current_person == person_num:
                    current_content.append(content)
                else:
                    if current_content:
                        merged_text = " ".join(current_content)
                        merged_blocks.append(f"<Person{current_person}>{merged_text}</Person{current_person}>")
                    current_person = person_num
                    current_content = [content]

            if current_content:
                merged_text = " ".join(current_content)
                merged_blocks.append(f"<Person{current_person}>{merged_text}</Person{current_person}>")

            return "\n".join(merged_blocks)

        except Exception as e:
            logger.error(f"Error fixing alternating tags: {str(e)}")
            # Best-effort fallback: return original transcript on any error
            return transcript

    def compose_prompt_params(
        self,
        config_conversation: ConversationConfigModel,
        image_file_paths: list[str] | None = None,
        image_path_keys: list[str] | None = None,
        input_texts: str = "",
    ) -> PromptParams:
        if image_file_paths is None:
            image_file_paths = []
        if image_path_keys is None:
            image_path_keys = []
        return PromptParams(
            conversation_style=config_conversation.conversation_style,
            roles_person1=config_conversation.roles_person1,
            roles_person2=config_conversation.roles_person2,
            dialogue_structure=config_conversation.dialogue_structure,
            podcast_name=config_conversation.podcast_name,
            podcast_tagline=config_conversation.podcast_tagline,
            output_language=config_conversation.output_language,
            engagement_techniques=config_conversation.engagement_techniques,
        )


class ContentGenerator:
    def __init__(
        self,
        is_local: bool = False,
        model_name: str = DEFAULT_GEMINI_LLM,
        api_key_label: str = ApiKeyLabel.GEMINI,
        conversation_config: ConversationConfigModel | None = None,
    ):
        self.app_config = load_app_config_model()
        self.content_generator_config = self.app_config.content_generator

        if isinstance(conversation_config, ConversationConfigModel):
            self.config_conversation = conversation_config
        else:
            self.config_conversation = load_conversation_config_model(conversation_config)

        # Create output directories if they don't exist
        transcripts_dir = self.config_conversation.output_directories.transcripts

        if transcripts_dir and not os.path.exists(transcripts_dir):
            os.makedirs(transcripts_dir)

        self.is_local = is_local

        # Initialize LLM backend
        if not model_name:
            model_name = self.app_config.content_generator.llm_model
        if is_local:
            model_name = LOCAL_LLM_PLACEHOLDER

        llm_backend = LLMBackend(
            is_local=is_local,
            temperature=self.config_conversation.creativity,
            max_output_tokens=self.app_config.content_generator.max_output_tokens,
            model_name=model_name,
            api_key_label=api_key_label,
        )

        self.llm = llm_backend.llm

        self.standard_strategy = StandardContentStrategy(
            self.llm, self.content_generator_config, self.config_conversation
        )
        self.longform_strategy = LongFormContentStrategy(
            self.llm, self.content_generator_config, self.config_conversation
        )

    def __compose_prompt(self, num_images: int, longform: bool = False):
        """
        Compose the prompt for the LLM based on the content list.
        """
        cg = self.app_config.content_generator

        # Get base template and commit values
        base_template = cg.prompt_template
        base_commit = cg.prompt_commit

        # Modify template and commit for longform if configured
        if longform:
            template = cg.longform_prompt_template
            commit = cg.longform_prompt_commit
        else:
            template = base_template
            commit = base_commit

        prompt_template = hub.pull(f"{template}:{commit}")

        image_path_keys = []
        messages = []

        # Only add text content if input_text is not empty
        text_content = {
            "type": "text",
            "text": _DEFAULT_CONVERSATION_PROMPT,
        }
        messages.append(text_content)

        for i in range(num_images):
            key = f"image_path_{i}"
            image_content = {
                "image_url": {"url": f"{{{key}}}", "detail": _DEFAULT_IMAGE_DETAIL},
                "type": "image_url",
            }
            image_path_keys.append(key)
            messages.append(image_content)

        user_prompt_template = ChatPromptTemplate.from_messages(
            messages=[HumanMessagePromptTemplate.from_template(messages)]
        )
        user_instructions = self.config_conversation.user_instructions

        user_instructions = (
            "[[MAKE SURE TO FOLLOW THESE INSTRUCTIONS OVERRIDING THE PROMPT TEMPLATE IN CASE OF CONFLICT: "
            + user_instructions
            + "]]"
        )

        new_system_message = prompt_template.messages[0].prompt.template + "\n" + user_instructions

        # Compose messages from podcastfy_prompt_template and user_prompt_template
        combined_messages = (
            ChatPromptTemplate.from_messages([new_system_message]).messages + user_prompt_template.messages
        )

        # Create a new ChatPromptTemplate object with the combined messages
        composed_prompt_template = ChatPromptTemplate.from_messages(combined_messages)

        return composed_prompt_template, image_path_keys

    def generate_qa_content(
        self,
        input_texts: str = "",
        image_file_paths: list[str] | None = None,
        output_filepath: str | None = None,
        longform: bool = False,
    ) -> str:
        """
        Generate Q&A content based on input texts.

        Args:
            input_texts (str): Input texts to generate content from.
            image_file_paths (List[str]): List of image file paths.
            output_filepath (Optional[str]): Filepath to save the response content.
            is_local (bool): Whether to use a local LLM or not.
            model_name (str): Model name to use for generation.
            api_key_label (str): Environment variable name for API key.
            longform (bool): Whether to generate long-form content. Defaults to False.

        Returns:
            str: Generated conversation content

        Raises:
            ValueError: If strategy validation fails
            Exception: If there's an error in generating content.
        """
        try:
            # Get appropriate strategy
            strategy = self.longform_strategy if longform else self.standard_strategy

            # Validate inputs for chosen strategy
            strategy.validate(input_texts, image_file_paths or [])

            # Setup chain
            num_images = 0 if self.is_local else len(image_file_paths or [])
            self.prompt_template, image_path_keys = self.__compose_prompt(num_images, longform)
            self.parser = StrOutputParser()
            self.chain = self.prompt_template | self.llm | self.parser

            # Prepare parameters using strategy
            prompt_params = strategy.compose_prompt_params(
                self.config_conversation, image_file_paths or [], image_path_keys, input_texts
            )

            # Generate content using selected strategy
            self.response = strategy.generate(self.chain, input_texts, prompt_params)

            # Clean response using the same strategy
            self.response = strategy.clean(self.response, self.content_generator_config)

            logger.info("Content generated successfully")

            # Save output if requested
            if output_filepath:
                with open(output_filepath, "w") as file:
                    file.write(self.response)
                logger.info(f"Response content saved to {output_filepath}")

            return self.response

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise
