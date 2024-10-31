from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from model_config import load_gpt4o_mini_model

class QA:
    """
    A class to handle question-answering operations using a language model and retriever.
    """
    def __init__(self, context=None, chat_model=None):
        """
        Initialize the QA system.
        
        Args:
            retriever: Document retriever instance (optional)
            model: Language model instance (optional)
        """
        # Initialize model with default or provided model
        self.chat_model = chat_model or load_gpt4o_mini_model()
        
        # Store retriever
        self.context = context
        
        # Initialize prompt template
        self.template = """
        You are an AI Assistant that can give answer for any query asked based on the documents provided to answer.
        Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

        Question: {query}
        Context: {context}
        ### important: answer on Vietnamese language
        """
        
        # Create prompt and output parser
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.output_parser = StrOutputParser()
        
        # Setup the chain
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the processing chain."""
        self.chain = (
            {"context": self.context, "query": RunnablePassthrough()}
            | self.prompt
            | self.chat_model
            | self.output_parser
        )
    
    def answer(self, query: str) -> str:
        """
        Generate an answer for the given query.
        
        Args:
            query (str): The question to be answered
            
        Returns:
            str: The generated answer
            
        Raises:
            ValueError: If no retriever has been set
        """
        if self.context is None:
            raise ValueError("No retriever has been set. Please set a retriever using update_retriever() first.")
        
        try:
            response = self.chain.invoke(query)
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def __call__(self, query: str) -> str:
        """
        Allow the class to be called directly with a query.
        
        Args:
            query (str): The question to be answered
            
        Returns:
            str: The generated answer
        """
        return self.answer(query)