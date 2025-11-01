# base code from https://www.youtube.com/watch?v=N3vHJcHBS-w

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("AVA")

# Define prompts
@mcp.prompt()
def ava(user_name: str, user_title: str) -> str:
    """Global instructions for Artificial Virutal Assistant (AVA)"""
    with open("prompts/ava.md", "r") as file:
        template = file.read()
    return template.format(user_name=user_name, user_title=user_title)

# Define resources
@mcp.resource("resources://get_readme")
def get_readme() -> str:
    """Readme of the MCP server"""
    with open("resources/readme.md", "r") as file:
        return file.read()

# Define tools
@mcp.tool("summarize_text")
def summarize_text(text: str) -> str:
    """Summarize a given block of text into a concise preview.
    
    Args:
        text (str): The input text to summarize. Can be any length; 
                    long passages will be truncated in the output.
    
    Returns:
        dict: A dictionary containing:
            - 'summary' (str): The summarized or truncated version of the text.
    
    Raises:
        ValueError: If the input text is empty or not a string.
    
    Note:
        This is a lightweight demonstration tool designed for an MCP server.
        It does not use any external APIs or models.
        
        Behavior:
        - If the input text exceeds 100 characters, it returns the first 100 
          characters followed by an ellipsis ("...").
        - If the text is shorter, it returns the original text unchanged.
        
        This tool can serve as a placeholder for more advanced summarization
        (e.g., LLM-based or extractive summarization via external APIs).
    """

    return text[:100] + ("..." if len(text) > 100 else "")

if __name__ == "__main__":
    mcp.run(transport='stdio')