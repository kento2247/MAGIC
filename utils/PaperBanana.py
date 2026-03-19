import asyncio
import base64
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

from utils import config
from utils.agents.critic_agent import CriticAgent

# agents / utils はそのまま利用
from utils.agents.planner_agent import PlannerAgent
from utils.agents.polish_agent import PolishAgent
from utils.agents.retriever_agent import RetrieverAgent
from utils.agents.stylist_agent import StylistAgent
from utils.agents.vanilla_agent import VanillaAgent
from utils.agents.visualizer_agent import VisualizerAgent
from utils.paperviz_processor import PaperVizProcessor


class PaperBanana:
    def __init__(
        self,
        exp_mode: str = "demo_full",
        retrieval_setting: str = "auto",
        main_model_name: str = "",
        image_gen_model_name: str = "",
        aspect_ratio: str = "16:9",
        max_critic_rounds: int = 3,
    ):
        self.exp_mode = exp_mode
        self.retrieval_setting = retrieval_setting
        self.main_model_name = main_model_name
        self.image_gen_model_name = image_gen_model_name
        self.aspect_ratio = aspect_ratio
        self.max_critic_rounds = max_critic_rounds

        # ExpConfig
        self.exp_config = config.ExpConfig(
            dataset_name="Demo",
            split_name="demo",
            exp_mode=exp_mode,
            retrieval_setting=retrieval_setting,
            main_model_name=main_model_name,
            image_gen_model_name=image_gen_model_name,
            work_dir=Path("."),
        )

        # Processor
        self.processor = PaperVizProcessor(
            exp_config=self.exp_config,
            vanilla_agent=VanillaAgent(exp_config=self.exp_config),
            planner_agent=PlannerAgent(exp_config=self.exp_config),
            visualizer_agent=VisualizerAgent(exp_config=self.exp_config),
            stylist_agent=StylistAgent(exp_config=self.exp_config),
            critic_agent=CriticAgent(exp_config=self.exp_config),
            retriever_agent=RetrieverAgent(exp_config=self.exp_config),
            polish_agent=PolishAgent(exp_config=self.exp_config),
        )

    def _create_input(self, description: str, caption: str):
        return {
            "filename": "demo_input",
            "caption": caption,
            "content": description,
            "visual_intent": caption,
            "additional_info": {"rounded_ratio": self.aspect_ratio},
            "max_critic_rounds": self.max_critic_rounds,
        }

    async def _run(self, input_data):
        results = []
        async for result in self.processor.process_queries_batch(
            [input_data], max_concurrent=1, do_eval=False
        ):
            results.append(result)
        return results[0]

    def _extract_final_image(self, result):
        task_name = "diagram"

        # criticの最後を優先
        for round_idx in range(3, -1, -1):
            key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
            if key in result and result[key]:
                return self._b64_to_image(result[key])

        # fallback
        if self.exp_mode == "demo_full":
            key = f"target_{task_name}_stylist_desc0_base64_jpg"
        else:
            key = f"target_{task_name}_desc0_base64_jpg"

        if key in result and result[key]:
            return self._b64_to_image(result[key])

        return None

    def _b64_to_image(self, b64_str):
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))

    def _load_demo_image(self) -> Image.Image:
        candidate_paths = [Path(__file__).parent / "assets" / "sample.png"]
        demo_sample_root = Path(__file__).resolve().parent.parent / "demo_sample"
        if demo_sample_root.exists():
            candidate_paths.extend(sorted(demo_sample_root.glob("*/sample.png")))

        asset_root = Path(__file__).resolve().parent.parent / "assets"
        if asset_root.exists():
            candidate_paths.extend(sorted(asset_root.glob("*.png")))

        for path in candidate_paths:
            if path.exists():
                return Image.open(path)

        raise FileNotFoundError("No demo image available for PaperBanana demo mode.")

    def generate(self, description: str, caption: str, is_demo: bool) -> Image.Image:
        """
        Args:
            description: method / input text
            caption: figure caption

        Returns:
            PIL.Image
        """

        if is_demo:
            delay_minutes = 0.05
            print(f"📄 Description: {description}")
            print(f"🖼️ Caption: {caption}")
            time_to_wait = delay_minutes * 60
            print(f"⏳ Simulating processing time of {delay_minutes} minutes...")
            time.sleep(time_to_wait)
            image = self._load_demo_image()
            print("✅ Returning demo image.")

        else:
            input_data = self._create_input(description, caption)
            result = asyncio.run(self._run(input_data))
            image = self._extract_final_image(result)
            if image is None:
                raise RuntimeError("Failed to generate image")

        return image
