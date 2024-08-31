from typing import List, Dict, Any
from abc import ABC, abstractmethod
from openai import AzureOpenAI
from src.config.settings import get_settings
import logging

settings = get_settings()


class LLM(ABC):
    def __init__(self, model_name: str):
        """ 
        Args:
            model_name (str): The name of the model.
            model_provider (str): The provider or company behind the model
        """
        self.model_name = model_name
        self.client = None

    @abstractmethod
    def get_response(
        self, messages: List[str]
    ) -> str:
        """
        Args:
            messages (list):  A list of messages (conversation history).

        Returns:
            str: The generated response.
        """
        pass

    async def get_info(self) -> Dict[str, Any]:
        """
        Abstract method to return info of the LLM model.
        Returns:
            Dict[str, Any]: info.
        """
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
        }






class AzureDeployedGPT4oMiniLLM(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client : AzureOpenAI = AzureOpenAI(
            api_key=settings.GPT_4O_MINI_AZURE_API_KEY , 
            base_url=settings.GPT_4O_MINI_AZURE_TARGET_URI , 
            api_version=settings.GPT_4O_MINI_AZURE_API_VERSION
        )
    
    def get_response(
        self,
        messages: List[str], 
    ) -> str:
        try : 
            response = self.client.chat.completions.create(
                messages= messages , 
                model = self.model_name ,
            )
            return response.choices[0].message.content
        except Exception as e : 
            logging.error(f"Error occured in aget_response : {e}")



class CollectionLLM : 
    llm_collection : Dict[str , LLM]= {
        'gpt-4o-mini' : AzureDeployedGPT4oMiniLLM(model_name="gpt-4o-mini")
    }



