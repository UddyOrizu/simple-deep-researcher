# ---------------------------------------------------------------------------#
#                       STAGE 1: INTENT CLASSIFICATION                        #
# ---------------------------------------------------------------------------#
import logging
import os
from typing import Dict, Optional, Any
from security_pipeline.pipeline import PipelineContext, PipelineStage
import spacy
import numpy as np
import litellm
from litellm import completion


class QueryExpansionStage(PipelineStage):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configure LLM settings
        self.llm_config = {
            "model_name": os.environ.get("INTENT_MODEL", "ollama/gemma"),  # Default to ollama/gemma, but can be configured via env var
            "api_base": os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"),  # Default Ollama API base
            "use_llm": os.environ.get("USE_LLM_INTENT", "true").lower() == "true"  # Whether to use LLM-based classification
        }
        
       

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Process text using an ensemble of classification methods.
        
        This method runs all available classification methods (LLM, spaCy, keyword)
        and uses majority voting to determine the final intent classification.
        If a method fails, it's excluded from voting.
        """
        
        prompt =""" You are an expert research assistant tasked with expanding the following query to maximize the breadth and depth of information retrieval. Your goal is to:

                - Analyze the original query for ambiguity, multiple interpretations, and implicit assumptions.
                - Identify related entities, concepts, synonyms, and broader/narrower terms using knowledge graphs, semantic analysis, and contextual reasoning[1][3][7].
                - Consider different stakeholder perspectives, potential implications, and subtopics that may be relevant but not explicitly mentioned.
                - Think step-by-step, reasoning through the possible angles, contexts, and applications of the query (use a chain-of-thought approach)[2].
                - Generate a final, enhanced query that is comprehensive and nuanced.
                - Produce at least 10 additional, distinct sub-queries that cover alternative angles, specific facets, or related questions.

                **Format your output as follows:**

                1. **Original Query:**  
                [Insert user’s query here]

                2. **Reasoning and Analysis:**  
                - List possible interpretations and implications of the query.  
                - Identify key entities, related concepts, and potential ambiguities.  
                - Note relevant broader, narrower, or synonymous terms.

                3. **Final Enhanced Query:**  
                [A single, comprehensive query that incorporates multiple facets, synonyms, and related concepts.]

                4. **Expanded Sub-Queries:**  
                - Sub-query 1: [A specific angle or related question]  
                - Sub-query 2: [Another angle, possibly focusing on a different stakeholder or context]  
                - Sub-query 3: [A query exploring a narrower or broader aspect]  
                - ...  
                - Sub-query 10: [A query that explores a potential implication, controversy, or application]
                """